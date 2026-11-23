/** Offline, diagnostic-only information-set rollout evaluator.
 * Never uses the actual hidden hands to choose or score a move. It samples
 * complete worlds from only the actor's hand and public/private-to-actor facts.
 * The fully instantiated hands are used ONLY by the simulated game engine.
 */
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { StrategicEnvironment } from './strategic-env.js';
import { FoundationEnvironment, serveEnvironment } from './foundation-env.js';
import { beliefInformation, sampleBeliefWorlds } from '../strategic-beliefs.js';
import { createDeck, strength, validSave } from '../engine.js';

const kind = c => c.rank > 14 ? c.rank + 33 : c.suit * 13 + c.rank - 2;
const MAX_CHOICES = 12;
const MAX_WORLDS = 32;
const MAX_TOTAL_ROLLOUTS = 256;

export function checkAuditBudget({ indices, worlds, attempts, seed }) {
  if (!Array.isArray(indices) || indices.length < 1 || indices.length > MAX_CHOICES ||
      new Set(indices).size !== indices.length ||
      indices.some(i => !Number.isInteger(i) || i < 0) ||
      !Number.isInteger(worlds) || worlds < 4 || worlds > MAX_WORLDS ||
      indices.length * worlds > MAX_TOTAL_ROLLOUTS ||
      !Number.isInteger(attempts) || attempts < worlds || attempts > 4096 ||
      !Number.isSafeInteger(seed) || seed < 0 || seed > 4294967295) {
    throw new Error('Invalid bounded counterfactual audit budget');
  }
}

/** Preserve real actor card IDs and all public played IDs. Never touch true
 * opponent hands to construct possible worlds. The belief sampler uses card
 * types; here each type is mapped to an unused physical ID from the 3 decks.
 */
export function instantiateBeliefWorld(game, sampledHands) {
  const actor = game.turn;
  const locked = new Set([
    ...game.hands[actor].map(c => c.id),
    ...game.played.map(c => c.id),
  ]);
  if (locked.size !== game.hands[actor].length + game.played.length) {
    throw new Error('Invalid visible card identities');
  }
  const unused = new Map();
  for (const card of createDeck()) {
    if (!locked.has(card.id)) {
      const k = kind(card);
      if (!unused.has(k)) unused.set(k, []);
      unused.get(k).push(card);
    }
  }
  const { hands, exchanges, ...state } = game;
  const result = structuredClone(state);
  result.hands = sampledHands.map((hand, seat) => seat === actor ? structuredClone(hands[seat])
    : hand.map(card => {
      const choices = unused.get(kind(card));
      if (!choices?.length) throw new Error('Belief sample violates physical card conservation');
      return choices.pop();
    }));
  const cardsBySeat = result.hands.map((hand, seat) =>
    [...hand, ...result.played.filter(card => card.player === seat)]);
  result.exchanges = exchanges.map(({ from, to, tribute, returned }) => {
    // Bind exchange records to possible-world cards, not the real hidden transfer IDs.
    const known = actor === from || actor === to;
    const returns = returned ? cardsBySeat[from].filter(c => !known || kind(c) === kind(returned)) : [null];
    const tributes = cardsBySeat[to].filter(c => c.rank === tribute.rank &&
      (!known || kind(c) === kind(tribute)) && (actor !== to || returned || c.id === tribute.id));
    for (const candidate of tributes) {
      const higher = c => strength(c.rank, game.level) > strength(candidate.rank, game.level) ||
        c.rank === candidate.rank && c.suit < candidate.suit;
      const replacement = returns.find(c => cardsBySeat[from].every(d => d === c || !higher(d)));
      if (replacement !== undefined) return {from, to, tribute: candidate, returned: replacement};
    }
    throw new Error('Belief sample violates tribute or return constraints');
  });
  if (!validSave(result)) throw new Error('Counterfactual world failed game validation');
  return result;
}

function terminalScore(game, team) {
  if (!game.over) throw new Error('Incomplete counterfactual rollout');
  if (game.result.winner === null) return 0.5;
  return Number(game.result.winner === team);
}

function runContinuation(source, action, maxSteps) {
  const env = new FoundationEnvironment(maxSteps);
  env.game = structuredClone(source);
  env.actions = [action];
  env.takeStep({ op: 'step', index: 0 });
  while (!env.game.over) {
    env.takeStep({ op: 'heuristic' });
  }
  return terminalScore(env.game, source.turn % 2);
}

export function pairedStatistics(scores, chosenColumn, column) {
  if (!scores.length) return null;
  const n = scores.length;
  const values = scores.map(row => row[column]);
  const average = values.reduce((a, b) => a + b, 0) / n;
  const diffs = scores.map(row => row[column] - row[chosenColumn]);
  const gain = diffs.reduce((a, b) => a + b, 0) / n;
  const variance = n > 1 ? diffs.reduce((a, d) => a + (d - gain) ** 2, 0) / (n - 1) : 0;
  const standardError = Math.sqrt(variance / n);
  return { score: average, pairedDeltaVsChosen: gain, pairedStdError: standardError,
    paired95Approx: [Math.max(-1, gain - 1.96 * standardError), Math.min(1, gain + 1.96 * standardError)] };
}

export class CounterfactualEnvironment extends StrategicEnvironment {
  handle(request) {
    if (request.op !== 'counterfactual_audit') return super.handle(request);
    const { indices, worlds, attempts = Math.min(4096, worlds * 16), seed, maxSteps = 3000 } = request;
    checkAuditBudget({ indices, worlds, attempts, seed });
    if (!this.game || this.game.over || !Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > 6000 ||
        indices.some(i => i >= this.actions.length)) {
      throw new Error('Counterfactual audit requires a live decision and legal action indices');
    }
    const actor = this.game.turn;
    const info = beliefInformation(this.game);
    const sampled = sampleBeliefWorlds(info, { samples: worlds, attempts, seed });
    // Always report insufficient samples, never silently use actual hidden hands.
    const result = { method: 'paired-belief-rollouts-v1', actor,
      candidates: indices.map(index => {
        const {kind, ids, target} = this.actions[index];
        return {index, kind, ids, target};
      }),
      sampledWorlds: sampled.worlds.length, attempts: sampled.tried,
      requestedWorlds: worlds, continuationPolicy: 'engine-heuristic-all-seats',
      conditioning: 'acting player private cards and public history only',
      caveat: 'Heuristic continuation and approximate beliefs; not proven causal optimality',
    };
    if (sampled.worlds.length < 4) {
      return { ...result, status: 'insufficient-belief-worlds', estimates: null };
    }
    const outcomeRows = [];
    for (const sampledHands of sampled.worlds) {
      const world = instantiateBeliefWorld(this.game, sampledHands);
      const row = [];
      for (const index of indices) {
        try {
          row.push(runContinuation(world, this.actions[index], maxSteps));
        } catch (error) {
          // An incomplete simulation is not assigned a fabricated outcome.
          return { ...result, status: 'rollout-failed', error: String(error.message), estimates: null };
        }
      }
      outcomeRows.push(row);
    }
    const estimates = indices.map((index, col) => ({ index, ...pairedStatistics(outcomeRows, 0, col) }));
    return { ...result, status: 'ok', estimates, evaluatedRollouts: outcomeRows.length * indices.length };
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  await serveEnvironment(new CounterfactualEnvironment());
}
