import readline from 'node:readline';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { newGame, computerTurn, validSave, publicPolicyHistory } from '../engine.js';
import { recordPolicyEvent } from '../policy.js';
import { ENCODING_VERSION, OBSERVATION_SIZE, ACTION_SIZE, HISTORY_SIZE,
  observeFoundation, legalFoundationActions, encodeFoundationHistory, applyFoundationAction } from '../foundation-policy.js';

export class FoundationEnvironment {
  constructor(maxActions = Number(process.env.RL_MAX_ACTIONS || 6000)) {
    if (!Number.isInteger(maxActions) || maxActions < 1) throw new Error('Invalid action limit');
    this.maxActions = maxActions;
    this.game = null;
    this.actions = [];
    this.historyOffset = 0;
    this.moves = 0;
  }

  state() {
    const game = this.game;
    const events = encodeFoundationHistory(game, 0, this.historyOffset);
    this.historyOffset = publicPolicyHistory(game).length;
    this.actions = game.over ? [] : legalFoundationActions(game);
    return {
      terminal: game.over, player: game.turn, phase: game.phase, result: game.over ? game.result : null,
      observation: game.over ? Array(OBSERVATION_SIZE).fill(0) : observeFoundation(game),
      actions: this.actions, events,
    };
  }

  takeStep(request) {
    const game = this.game;
    if (!game || game.over) throw new Error('Reset before taking actions');
    if (this.moves >= this.maxActions) throw new Error('Episode action limit exceeded');
    const player = game.turn;
    const previousHistory = game.history.length;
    const phase = game.phase;
    let action;
    if (request.op === 'heuristic') computerTurn(game);
    else {
      if (!Number.isInteger(request.index) || request.index < 0 || request.index >= this.actions.length) throw new Error('Invalid legal action index');
      action = this.actions[request.index];
      applyFoundationAction(game, action);
    }
    if (action?.kind !== 'ask') recordPolicyEvent(game, player, previousHistory, phase);
    this.moves++;
    if (process.env.RL_CHECK_STATES === '1' && !validSave(game)) throw new Error('Invalid state after action');
  }

  handle(request) {
    const game = this.game;
    if (request.op === 'hello') return { encodingVersion: ENCODING_VERSION, observationSize: OBSERVATION_SIZE, actionSize: ACTION_SIZE, historySize: HISTORY_SIZE };
    if (request.op === 'reset') {
      if (!Number.isInteger(request.seed) || request.seed < 0 || request.seed > 4294967295) throw new Error('seed must be uint32');
      if (request.level !== undefined && (!Number.isInteger(request.level) || request.level < 2 || request.level > 13)) throw new Error('level must be 2..13');
      if (request.startingSeat !== undefined && (!Number.isInteger(request.startingSeat) || request.startingSeat < 0 || request.startingSeat > 5)) throw new Error('startingSeat must be 0..5');
      let seed = request.seed;
      const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
      for (let i = 0; i < 4; i++) random();
      let previous;
      if (request.continue) {
        if (!game?.over) throw new Error('Cannot continue an unfinished round');
        previous = game.result.champion ? null : game;
      } else {
        const level = request.level ?? 2 + Math.floor(random() * 12);
        const seat = request.startingSeat ?? Math.floor(random() * 6);
        previous = { round: 0, levels: [level, level], dealer: seat % 2, order: [seat] };
      }
      this.game = newGame(previous, random);
      this.historyOffset = 0;
      this.moves = 0;
      if (!validSave(this.game)) throw new Error('Invalid reset state');
      return this.state();
    }
    if (request.op === 'step' || request.op === 'heuristic') {
      this.takeStep(request);
      return this.state();
    }
    if (request.op === 'close') return { closed: true };
    throw new Error('Unknown environment operation');
  }
}

export async function serveEnvironment(environment) {
  const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of lines) {
    try {
      const request = JSON.parse(line);
      process.stdout.write(`${JSON.stringify(environment.handle(request))}\n`);
      if (request.op === 'close') break;
    } catch (error) {
      process.stdout.write(`${JSON.stringify({ error: error.message })}\n`);
    }
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  await serveEnvironment(new FoundationEnvironment());
}
