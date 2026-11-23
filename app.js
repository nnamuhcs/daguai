import { SUITS, NAMES, rankName, sortHand, newGame, classify, beats, playCards, passTurn, suggestion, computerTurn, validSave, canAsk, askCards, returnCard } from './engine.js';
import { observeForPolicy, legalPolicyActions, recordPolicyEvent } from './policy.js';
import { legalFoundationActions, encodeFoundationHistory, applyFoundationAction } from './foundation-policy.js';
import { strategicFeatures } from './strategic-policy.js';

const $ = id => document.getElementById(id);
const STORAGE_KEY = 'daguai-table-v2';
const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
let game = newGame();
let selected = new Set();
let auto = false;
let timer;
let toastTimer;
let sortedBySuit = false;
let saveFailed = false;
let paused = false;
let speed = 1200;
let drag = null;
let dragFrame;
let modelEnabled = false;
let modelReturns = false;
let modelEncoding = 1;
let modelAbort = null;
let modelGeneration = 0;

function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 3200);
}

try {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved) {
    const parsed = JSON.parse(saved);
    if (!validSave(parsed)) throw new Error('存档格式无效');
    game = parsed;
    if (game.direction === undefined) {
      if (game.phase === 'playing' && game.target) {
        game.direction = 1;
        game.nextDirection = -1;
        toast('已续局：本轮沿用旧顺序，轮结束后切换逆时针。');
      } else {
        game.direction = -1;
        toast('已续局：现在按逆时针出牌，下位玩家接风。');
      }
    } else {
      toast('已回到上次的牌桌，继续打吧。');
    }
  }
} catch (error) {
  console.error('读取牌局失败', error);
  toast('本地存档无法读取，已新开一局。');
}

function save() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(game));
    $('save-status').textContent = '✓ 牌局已自动保存在本机';
    saveFailed = false;
  } catch (error) {
    console.error('保存牌局失败', error);
    $('save-status').textContent = '无法保存，刷新后牌局将丢失';
    if (!saveFailed) toast('浏览器无法保存牌局，本局仍可继续。');
    saveFailed = true;
  }
}

function cardHTML(card, interactive = false) {
  const wild = card.rank > 14;
  const red = card.suit === 1 || card.suit === 3;
  const chosen = selected.has(card.id);
  const label = wild ? rankName(card.rank) : `${SUITS[card.suit]} ${rankName(card.rank)}`;
  const tribute = interactive && game.phase === 'returning' && game.exchanges[game.returnIndex]?.tribute.id === card.id;
  const artwork = wild ? `<img class="joker-art" src="/joker-${card.rank === 20 ? 'big' : 'small'}.svg" alt="" aria-hidden="true" draggable="false">` : `<span class="card-center">${SUITS[card.suit]}&#xfe0e;</span>`;
  const tag = interactive ? 'button' : 'div';
  return `<${tag} class="card ${red ? 'red' : ''} ${wild ? 'wild' : ''} ${wild && card.rank === 19 ? 'small-joker' : ''} ${tribute ? 'tribute' : ''} ${interactive && chosen ? 'selected' : ''}" ${interactive ? `type="button" data-card="${card.id}" aria-label="${escape(label)}${card.rank === game.level ? ' 级牌' : ''}${tribute ? ' 刚收到的贡牌，不可原样退回' : ''}" aria-pressed="${chosen}"` : `aria-label="${escape(label)}"`}><span class="card-corner">${rankName(card.rank)}${wild ? '' : `<small>${SUITS[card.suit]}&#xfe0e;</small>`}</span>${artwork}${card.rank === game.level ? '<span class="trump-mark">级</span>' : ''}</${tag}>`;
}

function patternLabel(type) {
  if (![0, 3, 4, 6].includes(type.category)) return type.name;
  const value = type.values[0];
  const rank = value === 18 ? `${rankName(game.level)}（级牌）` : value === 21 ? '大怪' : value === 20 ? '小怪' : rankName(value);
  return `${type.name} ${rank}`;
}

function render() {
  endDrag();
  $('round-label').textContent = `第 ${String(game.round).padStart(2, '0')} 局`;
  $('level-label').textContent = `当前级牌 ${rankName(game.level)} · ${game.nextDirection ? '旧轮顺序，结束后逆时针' : '逆时针'}`;
  $('hand-count').textContent = `${game.hands[0].length} 张`;
  const previousRound = !game.target && game.history.length > 0;
  const roundStart = game.history.findLastIndex(move => move.lead);
  const roundMoves = game.phase === 'returning' ? [] : game.history.slice(Math.max(0, roundStart));
  const finishedPlayers = game.order.filter(i => game.hands[i].length === 0);
  const finishName = rank => rank === 0 ? '头家' : `第 ${rank + 1} 名`;
  $('finish-strip').innerHTML = `<strong class="finish-strip-title">出完顺序</strong>${finishedPlayers.length
    ? finishedPlayers.map((i, rank) => `<span class="finish-chip ${rank === 0 ? 'first-finisher' : ''}" aria-label="${NAMES[i]}已出完，${finishName(rank)}"><b>${rank + 1}</b> ${NAMES[i]}<small>${rank === 0 ? '头家' : '已出完'}</small></span>`).join('')
    : '<span class="no-finishers">还没人出完</span>'}<span class="remaining-count">未出完 ${6 - finishedPlayers.length} 人</span>`;
  $('seats').innerHTML = NAMES.map((name, i) => {
    const rank = game.order.indexOf(i);
    const action = game.actions[i];
    const reported = game.reports[i]?.answer;
    const countLabel = i === 0 || game.over ? `<strong>${game.hands[i].length}</strong> 张` : typeof reported === 'number' ? `<strong>${reported}</strong> 张 · 报数` : '<span class="unreported">未报数 · 电脑</span>';
    const finished = finishedPlayers.includes(i);
    const trapped = game.over && !finished;
    const rankLabel = finished
      ? `<div class="finish-status"><strong>已出完</strong><span>${finishName(rank)}</span></div>`
      : trapped ? `<strong class="trapped-status">被关住</strong> · ${game.hands[i].length} 张` : countLabel;
    const lastPlay = roundMoves.findLast(move => move.player === i);
    const holding = !!game.target && game.lastPlayer === i;
    const passing = !finished && action && action.cards.length === 0;
    const active = !game.over && !finished && game.turn === i;
    const played = lastPlay
      ? `<div class="seat-play ${previousRound ? 'previous-round' : ''} ${holding ? 'winning-play' : ''}" ${lastPlay.reason ? `title="${escape(lastPlay.reason)}"` : ''} aria-label="${name}${previousRound ? '上一轮' : '本轮'}出牌：${patternLabel(lastPlay.type)}"><div class="play-caption">${holding ? '<strong>当前要压</strong>' : previousRound ? '上一轮' : '本轮已出'} · ${patternLabel(lastPlay.type)}${passing ? '<span class="pass-label">不出</span>' : ''}</div><div class="seat-cards">${lastPlay.cards.map(c => cardHTML(c)).join('')}</div></div>`
      : `<div class="seat-play empty-play">${finished ? '已出完，不再出牌' : trapped ? '本局被关住' : passing ? '<span class="pass-label">不出</span>' : active ? '等待出牌' : '尚未出牌'}</div>`;
    return `<div class="seat seat-${i} ${i % 2 ? 'orange' : 'teal'} ${active ? 'active' : ''} ${holding ? 'holding' : ''} ${finished ? `finished ${rank === 0 ? 'head-finisher' : ''}` : ''} ${trapped ? 'trapped' : ''}" ${active ? 'aria-current="true"' : ''}><div class="player-chip"><span class="avatar">${finished ? '✓' : ['你', '诚', '满', '周', '岚', '林'][i]}</span><div class="player-info"><div class="seat-name">${name}<small>${i % 2 ? '对手' : '同队'}</small></div><div class="seat-count">${rankLabel}</div></div>${active ? `<span class="turn-badge">${paused ? '待出牌' : i === 0 && !auto ? '轮到你' : '出牌中'}</span>` : ''}</div>${played}</div>`;
  }).join('');
  const lastMove = game.history.at(-1);
  $('trick').innerHTML = game.phase === 'returning'
    ? '<div class="trick-empty">礼尚往来，进贡还贡<small>全部还贡完成后，上局头家领出</small></div>'
    : game.target
    ? `<div class="trick-caption"><strong>${NAMES[game.lastPlayer]}</strong> 出了 ${patternLabel(game.target.type)} · 跟 ${game.target.cards.length} 张</div><div class="trick-cards">${game.target.cards.map(c => cardHTML(c)).join('')}</div>`
    : lastMove
      ? `<div class="trick-caption completed-caption">上一轮结束 · ${NAMES[game.turn]}自由领出</div><div class="trick-cards completed-cards">${lastMove.cards.map(c => cardHTML(c)).join('')}</div>`
      : `<div class="trick-empty">${NAMES[game.turn]}先出牌<small>自由领出 · 1 / 2 / 3 / 5 张</small></div>`;
  $('turn-status').classList.toggle('my-turn', game.turn === 0 && !game.over);
  $('turn-status').textContent = paused && !game.over ? `已暂停 · 下一位是${NAMES[game.turn]}` : game.over ? '本局已结束' : game.phase === 'returning' ? `${NAMES[game.turn]}${game.turn === 0 && !auto ? '，选一张牌还贡' : '正在还贡…'}` : game.turn === 0 ? auto ? '你正在托管' : game.target ? '轮到你 · 选牌跟出，或不出' : '轮到你 · 自由领出' : `轮到${NAMES[game.turn]}出牌`;
  const exchange = game.exchanges[game.returnIndex];
  $('tribute-banner').hidden = game.phase !== 'returning';
  if (game.phase === 'returning') {
    $('tribute-banner').innerHTML = `<strong>${NAMES[exchange.to]}收到${NAMES[exchange.from]}的${rankName(exchange.tribute.rank)}。</strong> ${exchange.to === 0 ? '请选择一张牌还贡，不能原样退回贡牌。' : '等待电脑选择还贡。'}<br><small>已完成 ${game.returnIndex} / ${game.exchanges.length} 次还贡；完成后每人恢复 27 张。</small>`;
  }
  const hand = sortedBySuit ? [...game.hands[0]].sort((a, b) => a.suit - b.suit || b.rank - a.rank) : sortHand(game.hands[0], game.level);
  $('hand').innerHTML = hand.length ? hand.map(c => cardHTML(c, true)).join('') : '<span class="empty-hand">你的牌已出完，好好看同伴表演。</span>';
  $('scoreboard').innerHTML = [0, 1].map(team => {
    const members = [0, 1, 2, 3, 4, 5].filter(i => i % 2 === team);
    const done = members.filter(i => finishedPlayers.includes(i)).length;
    return `<div class="team-row ${team ? 'orange' : ''}"><div class="team-info"><span class="team-dot ${team ? 'orange' : 'teal'}"></span>${team ? '橙队' : '青队'}${team === game.dealer ? '<small>庄家</small>' : ''}<p class="team-members">${members.map(i => `<span class="${finishedPlayers.includes(i) ? 'member-finished' : ''}">${finishedPlayers.includes(i) ? '✓ ' : ''}${NAMES[i]}</span>`).join(' · ')}</p></div><div class="team-level">${game.levels[team] === 14 ? '✓' : rankName(game.levels[team])}<small>${game.levels[team] === 14 ? '通关' : '等级'}</small></div><div class="team-progress"><strong>已出完 ${done} / 3</strong><span aria-hidden="true">${'●'.repeat(done)}${'○'.repeat(3 - done)}</span></div></div>`;
  }).join('');
  $('logs').innerHTML = [...game.logs].reverse().slice(0, 15).map(message => `<li>${escape(message)}</li>`).join('');
  $('recent-plays').innerHTML = game.history.length ? game.history.slice(-8).reverse().map((move, index) => `<li><div class="history-heading"><strong class="${move.player % 2 ? 'orange-name' : 'teal-name'}">${NAMES[move.player]}</strong><span>${patternLabel(move.type)}</span><small>#${game.history.length - index}</small></div><div class="history-cards">${move.cards.map(c => cardHTML(c)).join('')}</div>${move.reason ? `<p class="history-reason">${escape(move.reason)}</p>` : ''}</li>`).join('') : '<li class="history-empty">出牌后，这里会保留具体牌面。</li>';
  scrollControls();
  controls();
}

function controls() {
  const myTurn = game.turn === 0 && !game.over && !auto;
  const cards = game.hands[0].filter(c => selected.has(c.id));
  const type = classify(cards, game.level);
  const returning = game.phase === 'returning';
  $('play-button').disabled = !myTurn || (returning ? cards.length !== 1 || cards[0].id === game.exchanges[game.returnIndex].tribute.id : !beats(type, game.target?.type));
  $('play-button').innerHTML = game.over ? '查看结算' : returning ? '确认还贡' : '出牌 <span>→</span>';
  if (game.over) $('play-button').disabled = false;
  $('pass-button').disabled = !myTurn || returning || !game.target;
  $('hint-button').disabled = !myTurn;
  $('ask-button').disabled = !myTurn || returning || (game.target ? !canAsk(game, 0) : ![2, 4].some(i => canAsk(game, 0, i)));
  $('clear-button').disabled = !selected.size;
  $('selection-label').textContent = selected.size
    ? returning ? `已选 ${selected.size} 张 · 还贡须选一张非贡牌` : `已选 ${selected.size} 张 · ${type ? type.name + (beats(type, game.target?.type) ? '，可以出牌' : '，压不过桌面') : '暂未组成有效牌型'}`
    : game.over ? '本局已结束，点「查看结算」再来一局' : returning ? '请选一张还贡，金色描边的是收到的贡牌' : '点击手牌选择，点击已选牌取消';
}

function cancelModel() {
  modelGeneration++;
  modelAbort?.abort();
  modelAbort = null;
}

async function localAction(current, signal) {
  const actions = modelEncoding >= 2 ? legalFoundationActions(current) : legalPolicyActions(current);
  const extra = modelEncoding === 3 ? strategicFeatures(current, actions) : null;
  const observation = [...observeForPolicy(current), ...(extra?.observation ?? [])];
  if (extra) for (let i = 0; i < actions.length; i++)
    actions[i] = { ...actions[i], features: [...actions[i].features, ...extra.actionExtras[i]] };
  const history = modelEncoding >= 2 ? encodeFoundationHistory(current) : undefined;
  let best = -Infinity;
  let chosen = null;
  for (let offset = 0; offset < actions.length; offset += 512) {
    const batch = actions.slice(offset, offset + 512);
    const response = await fetch('/api/ai/score', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ observation, actions: batch.map(a => a.features), ...(history ? { history } : {}) }), signal,
    });
    const result = await response.json();
    if (!response.ok || result.error) throw new Error(result.error || `模型服务 HTTP ${response.status}`);
    if (!Array.isArray(result.scores) || result.scores.length !== batch.length || !result.scores.every(Number.isFinite)) {
      throw new Error('模型返回了无效评分');
    }
    result.scores.forEach((score, i) => {
      if (score > best) { best = score; chosen = batch[i]; }
    });
  }
  if (!chosen) throw new Error('模型没有选择合法动作');
  return chosen;
}

function schedule() {
  clearTimeout(timer);
  cancelModel();
  if (game.over || paused || game.turn === 0 && !auto || document.querySelector('dialog[open]')) return;
  const generation = modelGeneration;
  timer = setTimeout(async () => {
    const current = game;
    const player = game.turn;
    const historyLength = game.history.length;
    const phase = game.phase;
    let asked = false;
    try {
      if (modelEnabled && (phase === 'playing' || modelReturns && phase === 'returning')) {
        const controller = new AbortController();
        modelAbort = controller;
        let action;
        try {
          action = await localAction(current, controller.signal);
        } catch (error) {
          if (generation !== modelGeneration) return;
          console.error('本地模型推理失败', error);
          modelEnabled = false;
          $('ai-select').value = 'heuristic';
          $('ai-status').textContent = `本地模型失败：${error.message}。已切回规则电脑。`;
          toast('本地模型推理失败，已切回规则电脑。');
          schedule();
          return;
        }
        if (generation !== modelGeneration || current !== game || paused || document.querySelector('dialog[open]')) return;
        if (modelEncoding >= 2) {
          applyFoundationAction(game, action);
          asked = action.kind === 'ask';
        } else computerTurn(game, action.ids);
      } else computerTurn(game);
      if (!asked) recordPolicyEvent(game, player, historyLength, phase);
      selected.clear();
      update();
    } catch (error) {
      console.error('电脑行动失败', error);
      auto = false;
      $('auto-toggle').checked = false;
      controls();
      toast(`牌局暂停：${error.message}`);
    }
  }, speed);
}

function update() {
  render();
  save();
  if (game.over) showResult();
  else schedule();
}

function showResult() {
  const result = game.result;
  $('result-title').textContent = result.champion ? '一路通关！' : result.winner === null ? '势均力敌，这局平手' : result.winner === 0 ? result.sweep ? '青队包揽前三！' : '这一路，青队赢了！' : result.sweep ? '橙队包揽前三' : '橙队先行一步';
  $('result-message').textContent = result.message;
  $('ranking').innerHTML = game.order.map((player, index) => `<div class="rank-item ${player % 2 ? 'orange' : ''} ${index >= result.finishedCount ? 'trapped' : ''}"><small>${index >= result.finishedCount ? `被关住 · ${game.hands[player].length} 张` : index === 0 ? '头家' : `第 ${index + 1} 名`}</small>${NAMES[player]}</div>`).join('');
  $('next-button').textContent = result.champion ? '新的征程 · 从 2 开始 →' : '再来一局 →';
  if (!$('result-dialog').open) $('result-dialog').showModal();
}

function start(previous = null) {
  clearTimeout(timer);
  cancelModel();
  game = newGame(previous);
  selected.clear();
  sortedBySuit = false;
  $('sort-button').textContent = '⇅ 理牌';
  for (const dialog of document.querySelectorAll('dialog[open]')) dialog.close();
  update();
}

function updateSelectedCards() {
  for (const button of $('hand').querySelectorAll('[data-card]')) {
    const chosen = selected.has(Number(button.dataset.card));
    button.classList.toggle('selected', chosen);
    button.setAttribute('aria-pressed', String(chosen));
  }
  controls();
}

function endDrag() {
  cancelAnimationFrame(dragFrame);
  const pointer = drag?.pointer;
  drag = null;
  if (pointer !== undefined && $('hand').hasPointerCapture(pointer)) $('hand').releasePointerCapture(pointer);
}

function dragSelect(clientX) {
  if (!drag) return;
  const hand = $('hand');
  const bounds = hand.getBoundingClientRect();
  const x = clientX - bounds.left + hand.scrollLeft;
  let index = 0;
  for (let i = 0; i < drag.positions.length; i++) {
    if (x >= drag.positions[i]) index = i;
  }
  selected = new Set(drag.original);
  for (let i = Math.min(drag.anchor, index); i <= Math.max(drag.anchor, index); i++) {
    if (drag.adding) selected.add(drag.ids[i]);
    else selected.delete(drag.ids[i]);
  }
  updateSelectedCards();
}

function dragScroll() {
  cancelAnimationFrame(dragFrame);
  if (!drag || !drag.moved) return;
  const hand = $('hand');
  const bounds = hand.getBoundingClientRect();
  const delta = drag.x < bounds.left + 28 ? -9 : drag.x > bounds.right - 28 ? 9 : 0;
  const before = hand.scrollLeft;
  hand.scrollLeft += delta;
  if (hand.scrollLeft !== before) {
    dragSelect(drag.x);
    dragFrame = requestAnimationFrame(dragScroll);
  }
}

$('hand').addEventListener('pointerdown', event => {
  const button = event.target.closest('[data-card]');
  if (!button || event.button !== 0 || auto || game.over || game.phase !== 'playing') return;
  endDrag();
  const hand = $('hand');
  const buttons = [...hand.querySelectorAll('[data-card]')];
  const bounds = hand.getBoundingClientRect();
  const id = Number(button.dataset.card);
  drag = {
    pointer: event.pointerId, anchor: buttons.indexOf(button), original: new Set(selected),
    adding: !selected.has(id), ids: buttons.map(b => Number(b.dataset.card)),
    positions: buttons.map(b => b.getBoundingClientRect().left - bounds.left + hand.scrollLeft),
    x: event.clientX, startX: event.clientX, moved: false,
  };
  hand.setPointerCapture(event.pointerId);
  if (event.pointerType === 'mouse') event.preventDefault();
  selected.has(id) ? selected.delete(id) : selected.add(id);
  updateSelectedCards();
});
$('hand').addEventListener('pointermove', event => {
  if (!drag || event.pointerId !== drag.pointer) return;
  drag.x = event.clientX;
  if (Math.abs(event.clientX - drag.startX) > 6) drag.moved = true;
  if (drag.moved) {
    dragSelect(event.clientX);
    dragScroll();
  }
});
$('hand').addEventListener('pointerup', event => {
  if (!drag || event.pointerId !== drag.pointer) return;
  if (drag.moved) dragSelect(event.clientX);
  endDrag();
});
$('hand').addEventListener('pointercancel', () => {
  if (!drag) return;
  selected = new Set(drag.original);
  endDrag();
  updateSelectedCards();
});
$('hand').addEventListener('lostpointercapture', () => endDrag());
$('hand').addEventListener('click', event => {
  const button = event.target.closest('[data-card]');
  if (!button || auto || game.over) return;
  if (game.phase === 'playing' && event.detail > 0) return;
  const id = Number(button.dataset.card);
  if (game.phase === 'returning') {
    const chosen = selected.has(id);
    selected.clear();
    if (!chosen) selected.add(id);
    render();
    return;
  }
  selected.has(id) ? selected.delete(id) : selected.add(id);
  button.classList.toggle('selected', selected.has(id));
  button.setAttribute('aria-pressed', String(selected.has(id)));
  controls();
});
function scrollControls() {
  const hand = $('hand');
  $('hand-left').disabled = hand.scrollLeft <= 1;
  $('hand-right').disabled = hand.scrollLeft + hand.clientWidth >= hand.scrollWidth - 1;
}
$('hand').addEventListener('scroll', scrollControls, { passive: true });
$('hand-left').addEventListener('click', () => $('hand').scrollBy({ left: -180, behavior: 'smooth' }));
$('hand-right').addEventListener('click', () => $('hand').scrollBy({ left: 180, behavior: 'smooth' }));
$('play-button').addEventListener('click', () => {
  if (game.over) return showResult();
  try {
    const historyLength = game.history.length;
    const phase = game.phase;
    if (game.phase === 'returning') returnCard(game, 0, [...selected][0]);
    else playCards(game, 0, [...selected]);
    recordPolicyEvent(game, 0, historyLength, phase);
    selected.clear();
    update();
  } catch (error) { toast(error.message); }
});
$('pass-button').addEventListener('click', () => {
  try {
    const historyLength = game.history.length;
    passTurn(game, 0);
    recordPolicyEvent(game, 0, historyLength, 'playing');
    selected.clear();
    update();
  } catch (error) { toast(error.message); }
});
$('hint-button').addEventListener('click', () => {
  const cards = suggestion(game, 0);
  if (!cards) {
    selected.clear();
    render();
    toast(game.lastPlayer !== null && game.lastPlayer % 2 === 0 ? '同伴正领跑，建议不出，让一路。' : '没有能压过的牌，建议不出。');
    return;
  }
  selected = new Set(cards.map(c => c.id));
  render();
  $('hand').querySelector('.selected')?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
});
$('clear-button').addEventListener('click', () => { selected.clear(); render(); });
$('sort-button').addEventListener('click', () => {
  sortedBySuit = !sortedBySuit;
  $('sort-button').textContent = sortedBySuit ? '⇅ 按点数' : '⇅ 理牌';
  render();
});
$('auto-toggle').addEventListener('change', event => {
  auto = event.target.checked;
  selected.clear();
  render();
  schedule();
});
$('pause-button').addEventListener('click', () => {
  paused = !paused;
  $('pause-button').textContent = paused ? '继续牌局' : '暂停看牌';
  $('pause-button').setAttribute('aria-pressed', String(paused));
  render();
  schedule();
});
$('speed-select').addEventListener('change', event => {
  speed = Number(event.target.value);
  schedule();
});
function openDialog(id) {
  clearTimeout(timer);
  cancelModel();
  $(id).showModal();
}
$('rules-button').addEventListener('click', () => openDialog('rules-dialog'));
$('restart-button').addEventListener('click', () => openDialog('restart-dialog'));
$('ask-button').addEventListener('click', () => {
  if (game.target) return ask(game.lastPlayer);
  $('ask-choices').innerHTML = [2, 4].filter(i => canAsk(game, 0, i)).map(i => `<button class="action secondary" data-ask="${i}">${NAMES[i]}</button>`).join('');
  openDialog('ask-dialog');
});
function ask(other) {
  try {
    const answer = askCards(game, 0, other);
    toast(`${NAMES[answer.player]}：${answer.count === null ? '不报，超过 10 张。' : `${answer.count} 张。`}`);
    if ($('ask-dialog').open) $('ask-dialog').close();
    render();
    save();
  } catch (error) { toast(error.message); }
}
$('ask-choices').addEventListener('click', event => {
  const button = event.target.closest('[data-ask]');
  if (button) ask(Number(button.dataset.ask));
});
$('confirm-restart').addEventListener('click', () => start());
$('next-button').addEventListener('click', () => start(game.result.champion ? null : game));
document.querySelectorAll('[data-close]').forEach(button => button.addEventListener('click', () => $(button.dataset.close).close()));
document.querySelectorAll('dialog').forEach(dialog => {
  dialog.addEventListener('close', () => schedule());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });
});
document.addEventListener('keydown', event => {
  if (document.querySelector('dialog[open]') || event.target.matches('input,button') || event.repeat) return;
  if (event.key === 'Enter' && !$('play-button').disabled) $('play-button').click();
  if (event.key === 'Escape') { selected.clear(); render(); }
});

async function refreshModel(reload = false) {
  $('reload-model').disabled = true;
  if (reload) { clearTimeout(timer); cancelModel(); }
  try {
    const response = await fetch(reload ? '/api/ai/reload' : '/api/ai/status', { method: reload ? 'POST' : 'GET' });
    const status = await response.json();
    if (!response.ok) throw new Error(status.error || `模型服务 HTTP ${response.status}`);
    $('ai-select').querySelector('[value="local"]').disabled = !status.available;
    if (!status.available) {
      modelEnabled = false;
      $('ai-select').value = 'heuristic';
      $('ai-status').textContent = '未载入模型。训练后以 npm run start:model 启动，才可选择试验模型。';
      $('ai-status').title = status.error || '';
      return;
    }
    modelEncoding = status.encodingVersion ?? 1;
    if (![1, 2, 3].includes(modelEncoding)) throw new Error('模型编码版本不受支持');
    modelReturns = status.learnReturns === true;
    $('ai-status').title = '';
    $('ai-status').textContent = `本地权重：训练 ${status.games} 局，${modelReturns ? '含模型还贡' : '还贡仍用规则'}${modelEncoding >= 2 ? '、自主问牌和整局公共历史' : ''}${modelEncoding === 3 ? '、v3手牌结构（实验版）' : ''}。只影响电脑和托管，尚未证明强于规则电脑。`;
  } catch (error) {
    console.error('读取本地模型状态失败', error);
    modelEnabled = false;
    $('ai-select').value = 'heuristic';
    $('ai-status').textContent = `本地模型不可用：${error.message}。已切回规则电脑。`;
    if (reload) toast('重载本地权重失败，已切回规则电脑。');
  } finally {
    $('reload-model').disabled = false;
    if (reload) schedule();
  }
}
$('reload-model').addEventListener('click', () => refreshModel(true));
update();
refreshModel();
$('ai-select').addEventListener('change', event => {
  modelEnabled = event.target.value === 'local';
  if (modelEnabled) toast('已选择本地试验模型；短期训练不代表牌力更强。');
  schedule();
});
