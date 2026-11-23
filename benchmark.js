import { newGame, computerTurn, candidates, playCards, passTurn } from './engine.js';

function basicTurn(game) {
  if (game.phase === 'returning') return computerTurn(game);
  const hand = game.hands[game.turn];
  const moves = candidates(hand, game.level, game.target?.type);
  const finish = moves.find(m => m.cards.length === hand.length);
  if (finish) return playCards(game, game.turn, finish.cards.map(c => c.id));
  if (!moves.length || game.target && game.lastPlayer % 2 === game.turn % 2) return passTurn(game, game.turn);
  const cost = m => m.cards.reduce((sum, c) => sum + (c.rank > 14 ? 24 : c.rank === game.level ? 12 : c.rank / 10), 0) +
    m.type.category * 2 - (!game.target ? m.cards.length * 5 : 0);
  moves.sort((a, b) => cost(a) - cost(b));
  playCards(game, game.turn, moves[0].cards.map(c => c.id));
}

const totalSeeds = Number(process.env.SEEDS || 30);
const results = { advancedWins: 0, basicWins: 0, draws: 0, games: 0 };
const started = performance.now();
for (let n = 0; n < totalSeeds; n++) {
  for (const advanced of [0, 1]) {
    let seed = n + 67;
    const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    const game = newGame(null, random);
    let turns = 0;
    while (!game.over && turns++ < 1500) {
      if (game.turn % 2 === advanced) computerTurn(game);
      else basicTurn(game);
    }
    if (!game.over) throw new Error(`seed ${n}: game did not finish`);
    results.games++;
    if (game.result.winner === null) results.draws++;
    else if (game.result.winner === advanced) results.advancedWins++;
    else results.basicWins++;
  }
}
console.log(JSON.stringify({ ...results, seconds: Math.round((performance.now() - started) / 1000), note: 'Paired identical deals, teams swapped; this compares against a simple baseline, not expert humans.' }, null, 2));
