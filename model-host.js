import { spawn } from 'node:child_process';
import { access } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import readline from 'node:readline';
import { ENCODING_VERSION, OBSERVATION_SIZE, ACTION_SIZE } from './policy.js';
import { ENCODING_VERSION as FOUNDATION_VERSION, ACTION_SIZE as FOUNDATION_ACTION_SIZE, HISTORY_SIZE } from './foundation-policy.js';
import { manifest as STRATEGIC_SCHEMA } from './strategic-policy.js';

const root = fileURLToPath(new URL('.', import.meta.url));
export class ModelHost {
  constructor() {
    this.checkpoint = resolve(root, process.env.AI_MODEL || 'training/checkpoints/approved.pt');
    this.python = process.env.AI_PYTHON || resolve(root, '.venv/bin/python');
    if (process.env.AI_WORKER && !['legacy', 'foundation', 'strategic'].includes(process.env.AI_WORKER)) throw new Error('Unknown AI_WORKER');
    this.strategic = process.env.AI_WORKER === 'strategic';
    this.foundation = this.strategic || process.env.AI_WORKER === 'foundation';
    this.encodingVersion = this.strategic ? STRATEGIC_SCHEMA.encodingVersion
      : this.foundation ? FOUNDATION_VERSION : ENCODING_VERSION;
    this.observationSize = this.strategic ? STRATEGIC_SCHEMA.observationSize : OBSERVATION_SIZE;
    this.actionSize = this.strategic ? STRATEGIC_SCHEMA.actionSize
      : this.foundation ? FOUNDATION_ACTION_SIZE : ACTION_SIZE;
    this.historySize = this.foundation ? HISTORY_SIZE : null;
    this.child = null;
    this.pending = [];
    this.info = null;
    this.starting = null;
  }
  async start() {
    if (this.info) return this.info;
    if (this.starting) return this.starting;
    this.starting = this.launch();
    try {
      return await this.starting;
    } finally {
      this.starting = null;
    }
  }
  async launch() {
    await access(this.checkpoint);
    const args = this.strategic
      ? [resolve(root, 'training/serve_strategic.py'), '--checkpoint', this.checkpoint, '--device', process.env.AI_DEVICE || 'cpu']
      : this.foundation
      ? [resolve(root, 'training/foundation.py'), 'infer', '--checkpoint', this.checkpoint, '--device', process.env.AI_DEVICE || 'cpu']
      : [resolve(root, 'training/infer.py'), '--checkpoint', this.checkpoint, '--device', process.env.AI_DEVICE || 'cpu', '--stdio'];
    const child = spawn(this.python, args, { cwd: root, stdio: ['pipe', 'pipe', 'pipe'] });
    this.child = child;
    let diagnostic = '';
    child.stderr.on('data', data => { diagnostic = (diagnostic + data).slice(-3000); });
    child.on('error', error => this.fail(child, error));
    child.on('exit', code => this.fail(child, new Error(`Model process exited (${code}): ${diagnostic}`)));
    const lines = readline.createInterface({ input: child.stdout });
    lines.on('line', line => {
      if (this.child !== child) return;
      const request = this.pending.shift();
      if (!request) return this.fail(child, new Error('Unexpected model response'));
      clearTimeout(request.timer);
      try {
        const response = JSON.parse(line);
        if (response.error) request.reject(new Error(response.error));
        else request.resolve(response);
      } catch (error) {
        request.reject(error);
        this.fail(child, error);
      }
    });
    child.on('exit', () => lines.close());
    const info = await this.rpc({ op: 'info' });
    if (info.encodingVersion !== this.encodingVersion ||
      info.observationSize !== this.observationSize || info.actionSize !== this.actionSize ||
      this.foundation && info.historySize !== this.historySize) {
      const error = new Error('Checkpoint observation/action schema does not match the game');
      this.fail(child, error);
      throw error;
    }
    this.info = info;
    return info;
  }
  fail(child, error) {
    if (this.child !== child) return;
    this.child = null;
    this.info = null;
    for (const request of this.pending.splice(0)) {
      clearTimeout(request.timer);
      request.reject(error);
    }
    child.kill();
  }
  rpc(payload) {
    if (!this.child) return Promise.reject(new Error('Model is not running'));
    if (this.pending.length >= 16) return Promise.reject(new Error('Local model is busy; retry later'));
    const child = this.child;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.fail(child, new Error('Local model request timed out')), 30000);
      this.pending.push({ resolve, reject, timer });
      child.stdin.write(`${JSON.stringify(payload)}\n`, error => {
        if (error) this.fail(child, error);
      });
    });
  }
  async score(observation, actions, history) {
    await this.start();
    const response = await this.rpc({ op: 'score', observation, actions, ...(this.foundation ? { history } : {}) });
    if (!Array.isArray(response.scores) || response.scores.length !== actions.length || !response.scores.every(Number.isFinite)) {
      throw new Error('Invalid scores from local model');
    }
    return response.scores;
  }
  async reload() {
    if (this.starting) throw new Error('Model is still loading; retry after it finishes');
    this.close();
    return this.start();
  }
  close() {
    if (this.child) this.fail(this.child, new Error('Local model shut down'));
  }
}
