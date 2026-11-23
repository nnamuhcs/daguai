import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { AuditEnvironment } from './audit-env.js';
import { FoundationEnvironment } from './foundation-env.js';
import { validSave, candidates, publicPolicyHistory, sortHand } from '../engine.js';
import { observeFoundation, legalFoundationActions, encodeFoundationHistory } from '../foundation-policy.js';

export function actorInput(game) {
  return { observation: observeFoundation(game), history: encodeFoundationHistory(game),
    actions: legalFoundationActions(game) };
}

function signature(input) {
  return createHash('sha256').update(JSON.stringify(input)).digest('hex');
}

export function consistentWithQuestions(game) {
  if (game.phase !== 'playing') return true;
  for (const question of game.questions) {
    if (!question.target) continue;
    const laterCards = game.played.slice(question.playedAt).filter(c => c.player === question.player);
    const handAtQuestion = [...game.hands[question.player], ...laterCards];
    if (!candidates(handAtQuestion, game.level, question.target).length) return false;
  }
  return true;
}

export function consistentWithTribute(game) {
  for (const exchange of game.exchanges) {
    const original = [...game.hands[exchange.from], ...game.played.filter(c => c.player === exchange.from)]
      .filter(c => c.id !== exchange.returned?.id);
    original.push(exchange.tribute);
    if (original.length !== 27 || sortHand(original, game.level)[0].id !== exchange.tribute.id) return false;
  }
  return true;
}

export function swapHiddenWorld(game, random) {
  const actor = game.turn;
  const protectedIds = new Set(game.exchanges.flatMap(e => [e.tribute.id, e.returned?.id]));
  const available = game.hands.map((hand, seat) => seat === actor ? [] :
    hand.map((card, index) => ({ card, index })).filter(({ card }) => !protectedIds.has(card.id)));
  const seats = available.map((cards, seat) => cards.length ? seat : null).filter(seat => seat !== null);
  if (seats.length < 2) return null;
  for (let attempt = 0; attempt < 100; attempt++) {
    const first = seats[Math.floor(random() * seats.length)];
    const second = seats.filter(seat => seat !== first)[Math.floor(random() * (seats.length - 1))];
    const a = available[first][Math.floor(random() * available[first].length)];
    const b = available[second][Math.floor(random() * available[second].length)];
    if (a.card.rank === b.card.rank && a.card.suit === b.card.suit) continue;
    const altered = structuredClone(game);
    altered.hands[first][a.index] = b.card;
    altered.hands[second][b.index] = a.card;
    if (validSave(altered) && consistentWithQuestions(altered) && consistentWithTribute(altered)) return altered;
  }
  return null;
}

export function runPrivacyAudit({ worlds = 10000, seed = 10000000, positions = 200 } = {}) {
  if (!Number.isInteger(worlds) || worlds < 1 || !Number.isInteger(positions) || positions < 4) throw new Error('Invalid audit size');
  const env = new AuditEnvironment();
  const production = new FoundationEnvironment();
  assert.throws(() => production.handle({ op: 'snapshot' }), /Unknown environment operation/);
  let randomSeed = seed >>> 0;
  const random = () => ((randomSeed = (Math.imul(randomSeed, 1664525) + 1013904223) >>> 0) / 4294967296);
  const phases = ['fresh', 'mid', 'late', 'returning'];
  const coverage = Object.fromEntries(phases.map(phase => [phase, 0]));
  const fixtureDigests = [];
  let checked = 0;
  let skippedSwaps = 0;
  for (let index = 0; checked < worlds && index < positions; index++) {
    const phase = phases[index % phases.length];
    env.handle({ op: 'sample', seed: seed + index * 64, phase });
    const original = structuredClone(env.game);
    const input = actorInput(original);
    const inputSignature = signature(input);
    const copiesPermuted = JSON.parse(JSON.stringify(original, (key, value) =>
      key === 'id' && Number.isInteger(value) ? (value + 54) % 162 : value));
    assert(validSave(copiesPermuted));
    const semanticInput = state => {
      const { observation, history, actions } = actorInput(state);
      return { observation, history, actions: actions.map(({ kind, target, features }) => ({ kind, target, features })) };
    };
    assert.deepEqual(semanticInput(copiesPermuted), semanticInput(original), 'Physical deck-copy IDs affected actor features');
    const needed = Math.min(Math.ceil(worlds / positions), worlds - checked);
    let accepted = 0;
    for (let attempt = 0; accepted < needed && attempt < needed * 10; attempt++) {
      const other = swapHiddenWorld(original, random);
      if (!other) { skippedSwaps++; continue; }
      assert.deepEqual(publicPolicyHistory(other), publicPolicyHistory(original));
      assert.equal(signature(actorInput(other)), inputSignature, 'Actor input leaked an unknown hand');
      checked++;
      accepted++;
      coverage[phase]++;
    }
    fixtureDigests.push({ seed: seed + index * 64, phase, digest: inputSignature, accepted });
  }
  if (checked !== worlds) throw new Error(`Only checked ${checked}/${worlds} indistinguishable worlds`);
  const fresh = new AuditEnvironment();
  fresh.handle({ op: 'reset', seed });
  const levelOriginal = observeFoundation(fresh.game);
  const newLevel = fresh.game.level === 13 ? 2 : fresh.game.level + 1;
  const levelChanged = structuredClone(fresh.game);
  levelChanged.level = newLevel;
  levelChanged.levels[levelChanged.dealer] = newLevel;
  assert.notDeepEqual(observeFoundation(levelChanged).slice(396, 409), levelOriginal.slice(396, 409));
  return { schemaVersion: 1, worlds: checked, independentPositions: fixtureDigests.length,
    coverage, skippedSwaps, fixtureDigests, passed: true, productionSnapshotBlocked: true,
    trumpAlreadyEncoded: true, copyIdentityFeatureInvariance: true, automaticHighestTributeConstrained: true,
    scope: 'Own/public observation, full public history and exact legal action invariance',
    limitation: 'Multiple worlds per position; input invariance implies deterministic actor/critic and seeded sampler invariance, not calibrated beliefs' };
}

async function main() {
  const args = process.argv.slice(2);
  if (args.includes('--help')) {
    console.log('node training/privacy-audit.js [--worlds 10000] [--positions 200] [--seed 10000000] --report PATH');
    return;
  }
  const value = name => { const index = args.indexOf(name); return index < 0 ? undefined : args[index + 1]; };
  const report = value('--report');
  if (!report) throw new Error('--report is required');
  const result = runPrivacyAudit({ worlds: Number(value('--worlds') ?? 10000),
    positions: Number(value('--positions') ?? 200), seed: Number(value('--seed') ?? 10000000) });
  await mkdir(dirname(resolve(report)), { recursive: true });
  await writeFile(report, `${JSON.stringify(result, null, 2)}\n`, { flag: 'wx' });
  console.log(`Privacy input invariance passed for ${result.worlds} worlds / ${result.independentPositions} positions`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) await main();
