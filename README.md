# Daguai Web · 大怪路子

A local browser version of the six-player Shanghai card game. Play with five
computer players on two teams of three, with asking, tribute, return, and
round settlement. The default AI uses rules and does not need a trained model.

## Play locally

With Node.js 24 or newer:

```sh
git clone https://github.com/nnamuhcs/daguai.git ~/daguai-web
cd ~/daguai-web
npm start
```

Open [localhost:3000](http://localhost:3000). To run the game-engine tests,
use `npm test`. The browser tests also require `npm ci` and
`npx playwright install chromium`.

## AI training

The repository includes the Python/Node training simulator, PPO trainer,
full-team evaluation, local dashboards, and a replay viewer. The
[continuous-learning runbook](CONTINUOUS_LEARNING.md) covers Python setup,
CUDA, and monitoring. See [implementation status](IMPLEMENTATION_STATUS.md)
for what was measured and what remains unproven.

The learner evaluates three trained players against three original web AI
players on paired deals with sides swapped. It does not replace the browser
AI automatically. Model weights, optimizer checkpoints, private game
recordings, and local training output are not included.
