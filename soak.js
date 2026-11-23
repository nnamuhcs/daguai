import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { newGame, computerTurn, validSave, canAsk, publicCounts } from './engine.js';

const rounds = Number(process.argv[2] || 200);
if (!Number.isInteger(rounds) || rounds < 1) throw new Error('Rounds must be a positive integer');
let seed = 20261121;
const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const stats = {
  requested: rounds, completed: 0, wins: [0, 0], draws: 0, sweeps: [0, 0],
  trapped: { 1: 0, 2: 0, 3: 0 }, actions: 0, passes: 0, plays: 0,
  tributeCards: 0, returnedCards: 0, asking: 0, noReportAnswers: 0,
  dealerUpgrades: 0, dealerChanges: 0, championships: [0, 0],
  roads: { 1: 0, 2: 0, 3: 0, 5: 0 }, types: {}, maxActionsPerGame: 0,
  conservationChecks: 0, savedResumeChecks: 0, errors: [], games: [],
};
const durations = [];
const started = performance.now();
let game = newGame(null, random);
let actionContext;

try {
  for (let round = 0; round < rounds; round++) {
    const oldLevels = [...game.levels];
    const oldDealer = game.dealer;
    let actions = 0;
    stats.tributeCards += game.exchanges.length;
    while (!game.over) {
      assert.ok(actions++ < 1500, `Round ${round + 1} did not terminate`);
      actionContext = { round: round + 1, action: actions, player: game.turn, phase: game.phase, hands: game.hands.map(h => h.length) };
      const player = game.turn;
      const returning = game.phase === 'returning';
      const historyLength = game.history.length;
      const questionCount = game.questions.length;
      const askAllowed = canAsk(game, player);
      const leading = game.target === null;
      const counts = publicCounts(game);
      const t0 = performance.now();
      computerTurn(game);
      durations.push(performance.now() - t0);
      stats.actions++;
      if (returning) {
        stats.returnedCards++;
        assert.equal(game.exchanges[game.returnIndex - 1].to, player);
        assert.notEqual(game.exchanges[game.returnIndex - 1].tribute.id, game.exchanges[game.returnIndex - 1].returned.id);
        if (game.phase === 'playing') assert.ok(game.hands.every(h => h.length === 27));
      } else if (game.history.length > historyLength) {
        const move = game.history.at(-1);
        stats.plays++;
        stats.roads[move.cards.length]++;
        stats.types[move.type.name] = (stats.types[move.type.name] || 0) + 1;
      } else {
        stats.passes++;
      }
      for (const question of game.questions.slice(questionCount)) {
        assert.ok(askAllowed || leading && question.other % 2 === player % 2 && question.other !== player, 'Illegal question');
        assert.ok(counts[question.other] <= 10, 'AI asked unnecessarily above reporting threshold');
        stats.asking++;
        if (question.answer === null) stats.noReportAnswers++;
      }
      assert.ok(validSave(game), `Invalid state after round ${round + 1}, action ${actions}`);
      stats.conservationChecks++;
      if (returning || actions % 20 === 0) {
        const resumed = JSON.parse(JSON.stringify(game));
        assert.ok(validSave(resumed));
        assert.deepEqual(resumed.hands, game.hands);
        assert.equal(resumed.phase, game.phase);
        stats.savedResumeChecks++;
        game = resumed;
      }
    }
    stats.completed++;
    stats.maxActionsPerGame = Math.max(stats.maxActionsPerGame, actions);
    const { winner, trapped, champion } = game.result;
    if (winner === null) stats.draws++;
    else {
      stats.wins[winner]++;
      stats.trapped[trapped.length]++;
      if (trapped.length === 3) stats.sweeps[winner]++;
    }
    if (game.dealer !== oldDealer) stats.dealerChanges++;
    if (game.levels.some((n, i) => n > oldLevels[i])) stats.dealerUpgrades++;
    if (champion) stats.championships[game.levels.indexOf(14)]++;
    stats.games.push({ game: round + 1, winner, actions, trapped: trapped.length, finished: game.result.finishedCount, order: game.order, levels: game.levels });
    if ((round + 1) % 25 === 0) console.log(`Completed ${round + 1}/${rounds}`);
    if (round + 1 < rounds) game = newGame(champion ? null : game, random);
  }
} catch (error) {
  stats.errors.push({ message: error.message, context: actionContext, stack: error.stack });
  process.exitCode = 1;
}

durations.sort((a, b) => a - b);
const report = {
  ...stats,
  runtimeSeconds: Math.round((performance.now() - started) / 1000),
  averageActionsPerGame: Math.round(stats.actions / Math.max(stats.completed, 1) * 10) / 10,
  thinkingMs: {
    median: Math.round(durations[Math.floor(durations.length * .5)] || 0),
    p95: Math.round(durations[Math.floor(durations.length * .95)] || 0),
    max: Math.round(durations.at(-1) || 0),
  },
  limitation: 'All six seats use the same bounded-search AI. This is a functional and stability soak, not evidence of expert-level play.',
};
if (process.env.REPORT_PATH) await writeFile(process.env.REPORT_PATH, JSON.stringify(report, null, 2));
const { games, ...summary } = report;
console.log(JSON.stringify(summary, null, 2));
