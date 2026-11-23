import { beats, candidates, classify } from './engine.js';
import { beliefInformation, sampleBeliefWorlds } from './strategic-beliefs.js';
import { strategicFeatures } from './strategic-policy.js';

const closeCounts = new Set([1, 2, 3, 5]);

function sampleSeed(info) {
  let seed = (info.level * 6 + info.actor) >>> 0;
  for (const value of [...info.own, ...info.played.flat(), ...info.counts]) {
    seed = Math.imul(seed ^ (value + 1), 16777619);
  }
  return seed >>> 0;
}

export function chooseStrategicGoal(game, legal, scores, { samples = 12, attempts = 96 } = {}) {
  if (!Array.isArray(legal) || !legal.length || scores.length !== legal.length ||
      !Array.from(scores).every(Number.isFinite)) throw new Error('Goal selection requires aligned finite logits');
  const base = Array.from(scores).reduce((top, score, i) => score > scores[top] ? i : top, 0);
  const actor = game.turn;
  const info = beliefInformation(game);
  const rotated = Array.from({ length: 5 }, (_, i) =>
    (actor + 6 + (game.direction ?? -1) * (i + 1)) % 6).filter(seat => info.counts[seat] > 0);
  const allies = seat => seat % 2 === actor % 2;
  if (game.phase !== 'playing' || legal.length === 1 ||
      (info.own.length > 12 && !rotated.some(seat => closeCounts.has(info.counts[seat])))) {
    return { index: base, goal: 'policy', samples: 0, attempts: 0 };
  }
  const { actionExtras } = strategicFeatures(game, legal);
  const teamFinish = actionExtras.findIndex(extra => extra[13]);
  if (teamFinish !== -1) return { index: teamFinish, goal: 'complete-team', samples: 0, attempts: 0 };
  const { worlds, tried } = sampleBeliefWorlds(info, { samples, attempts, seed: sampleSeed(info) });
  if (worlds.length < Math.min(4, samples)) {
    return { index: base, goal: 'insufficient-evidence', samples: worlds.length, attempts: tried };
  }
  const byId = new Map(game.hands[actor].map(card => [card.id, card]));
  const moves = legal.map(action => action.kind === 'play' ?
    classify(action.ids.map(id => byId.get(id)), info.level) : null);
  const needsControl = moves.some((move, i) => move && actionExtras[i][16]);
  const profile = worlds.map(world => rotated.map(seat => ({
    seat,
    exit: classify(world[seat], info.level),
    // A bounded candidate generator gives only an approximate control signal.
    strength: allies(seat) || !needsControl ? [] : candidates(world[seat], info.level)
      .reduce((best, move) => {
        if (beats(move.type, best[move.type.count])) best[move.type.count] = move.type;
        return best;
      }, []),
  })));
  const current = game.target?.type ?? null;
  const holder = game.lastPlayer;
  const lastUnpassed = rotated.findLastIndex(seat => seat !== holder && !game.passed.includes(seat));
  const leadDirection = game.nextDirection ?? game.direction ?? -1;
  const passLead = !current ? null : info.counts[holder] > 0 ? holder :
    Array.from({ length: 6 }, (_, i) => (holder + 6 + leadDirection * (i + 1)) % 6)
      .find(seat => info.counts[seat] > 0);
  const passClosesTrick = Boolean(current && lastUnpassed === -1);
  // An earlier pass does not skip a seat while the trick is still open.
  const passingRows = rows => !current ? rows : passClosesTrick ?
    rows.filter(row => row.seat === passLead) : rows.slice(0, lastUnpassed + 1);
  const passTarget = passClosesTrick ? null : current;
  const currentOpportunity = profile.reduce((sum, rows) => sum + Number(passingRows(rows).some(row =>
    !allies(row.seat) && beats(row.exit, passTarget))), 0) / worlds.length;
  const intervention = moves.map(move => {
    let defense = 0, feed = 0, control = 0;
    for (const world of profile) {
      const target = move ?? passTarget;
      const next = move ? world : passingRows(world);
      const enemyHasExit = next.some(row => !allies(row.seat) && beats(row.exit, target));
      const allyHasExit = next.some(row => allies(row.seat) && beats(row.exit, target));
      defense += Number(enemyHasExit);
      feed += Number(allyHasExit && !enemyHasExit);
      control += Number(Boolean(move) && !next.some(row =>
        !allies(row.seat) && beats(row.strength[move.count], move)));
    }
    return { danger: defense / worlds.length, feed: feed / worlds.length, control: control / worlds.length };
  });
  const exitPath = (i, entry) => moves[i] && (entry[12] || entry[16] && intervention[i].control >= .65);
  const allyLeads = Boolean(current && allies(game.lastPlayer));
  const allyKeepsTempo = passLead != null && allies(passLead) && passLead !== actor;
  const allySoon = rotated.some(seat => allies(seat) && closeCounts.has(info.counts[seat]));
  let goal;
  if (currentOpportunity >= .35 && intervention.some(row => row.danger + .15 < currentOpportunity)) goal = 'defend';
  else if (actionExtras.some((entry, i) => exitPath(i, entry))) goal = allyLeads ? 'receive' : 'lead';
  else if (allyKeepsTempo || allySoon && intervention.some((row, i) =>
    legal[i].kind !== 'ask' && row.feed >= .2)) goal = 'feed';
  else goal = 'preserve';
  const adjustment = actionExtras.map((entry, i) => {
    const action = legal[i], { danger, feed, control } = intervention[i];
    if (action.kind === 'ask') return 0;
    const canFinish = entry[12];
    const route = Number(exitPath(i, entry)) * (canFinish ? .9 : control);
    const used = entry[3] + entry[4];
    let contribution;
    if (goal === 'defend') contribution = currentOpportunity - danger + route * .2;
    else if (goal === 'receive' || goal === 'lead') contribution = route - danger * .7;
    else if (goal === 'feed') contribution = feed - danger + Number(allyKeepsTempo && action.kind === 'pass') * .25;
    else contribution = -used * .6 - danger * .5;
    return Math.max(-.15, Math.min(.15, .15 * contribution));
  });
  const index = adjustment.reduce((top, bonus, i) => scores[i] + bonus > scores[top] + adjustment[top] ? i : top, 0);
  return {
    index, goal, samples: worlds.length, attempts: tried, currentOpportunity,
    selected: intervention[index], scoreAdjustment: adjustment[index], changedAction: index !== base,
  };
}
