// Explicit local opt-in: serve the *statistically promoted* PPO checkpoint.
// Never serves the exploratory research-best model and never auto-reloads a live game.
import { readFile, realpath } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { dirname, join, relative, resolve, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));
const runRoot = resolve(root, process.env.DAGUAI_LEARNER_ROOT || 'training/ppo/continuous');
const registry = JSON.parse(await readFile(join(runRoot, 'progress.json'), 'utf8'));
const champion = registry.champion;
if (!champion || champion.checkpoint === 'heuristic') {
  throw new Error('No trained champion passed the promotion gates. Browser remains heuristic.');
}
const target = await realpath(champion.checkpoint);
const allowed = await realpath(join(runRoot, 'champions'));
const relativePath = relative(allowed, target);
if (!relativePath || relativePath.startsWith('..') || isAbsolute(relativePath)) {
  throw new Error('Champion must be an immutable file inside the learner champion directory');
}
const data = await readFile(target);
const digest = createHash('sha256').update(data).digest('hex');
if (champion.sha256 !== digest) throw new Error('Champion checkpoint digest mismatch');
if (![2, 3].includes(champion.encodingVersion)) throw new Error('Unknown promoted checkpoint encoding');
console.log(`Serving explicitly selected Daguai champion ${digest.slice(0, 16)} (v${champion.encodingVersion})`);
const server = spawn(process.execPath, [join(root, 'server.js')], {
  cwd: root, stdio: 'inherit', env: {...process.env, AI_MODEL: target, HOST: '127.0.0.1',
    AI_WORKER: champion.encodingVersion === 3 ? 'strategic' : 'foundation'}
});
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => { if (!server.killed) server.kill(signal); });
}
server.on('exit', (code, signal) => { process.exitCode = code ?? (signal ? 1 : 0); });
