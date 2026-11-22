import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
};

async function serve(dir) {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://127.0.0.1');
      let pathname = decodeURIComponent(url.pathname);
      if (pathname.endsWith('/')) pathname += 'index.html';
      const path = normalize(join(dir, pathname));
      const rootWithSep = dir.endsWith('/') ? dir : `${dir}/`;
      if (path !== dir && !path.startsWith(rootWithSep)) {
        res.writeHead(403);
        res.end();
        return;
      }
      const data = await readFile(path);
      const type = MIME[extname(path)] || 'application/octet-stream';
      res.writeHead(200, { 'content-type': type, 'cache-control': 'no-store' });
      res.end(data);
    } catch {
      res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
      res.end('not found');
    }
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();
  return {
    url: `http://127.0.0.1:${port}/index.html`,
    close: () => new Promise((resolve) => server.close(resolve)),
  };
}

async function launchBrowser(chromium) {
  const options = { headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] };
  try {
    return await chromium.launch(options);
  } catch {
    return await chromium.launch({ ...options, channel: 'chrome' });
  }
}

test('A8 headless Chromium loads a first field frame at 1100x800 with no console errors', { skip: process.env.WAP_BROWSER_SMOKE === '1' ? false : 'set WAP_BROWSER_SMOKE=1 to run the headless Chromium smoke' }, async () => {
  let chromium;
  try {
    ({ chromium } = await import('playwright'));
  } catch (err) {
    assert.fail(`Playwright is required when WAP_BROWSER_SMOKE=1 (${err instanceof Error ? err.message : err})`);
  }
  const hosted = await serve(root);
  const browser = await launchBrowser(chromium);
  const errors = [];
  const failed = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 800 }, deviceScaleFactor: 1 });
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    page.on('pageerror', (err) => errors.push(String(err)));
    page.on('response', (res) => {
      if (res.status() < 400) return;
      const path = new URL(res.url()).pathname;
      failed.push(`${res.status()} ${path}`);
    });
    await page.goto(hosted.url, { waitUntil: 'load', timeout: 20000 });
    await page.waitForFunction(() => {
      const canvas = document.querySelector('#field-canvas');
      const state = document.querySelector('#field-state');
      return canvas && Number(canvas.dataset.frame) >= 1 && state && state.dataset.state !== 'checking';
    }, null, { timeout: 15000 });

    const box = await page.locator('#field-canvas').boundingBox();
    assert.ok(box, 'field canvas has a box');
    assert.ok(box.height > 100, `field canvas height ${box.height}`);
    assert.ok(box.width > 180, `field canvas width ${box.width}`);
    assert.ok(box.y >= 0 && box.y + box.height <= 801, `field canvas y ${box.y} h ${box.height}`);
    assert.ok(box.x >= 0 && box.x + box.width <= 1101, `field canvas x ${box.x} w ${box.width}`);

    for (const id of ['#panel-channel', '#panel-signal', '#constellation', '#ber-plot', '#spectrum', '#channel-plot-0']) {
      const item = await page.locator(id).boundingBox();
      assert.ok(item && item.height > 24, `${id} height ${item?.height}`);
      assert.ok(item.y + item.height <= 801, `${id} extends past 800`);
    }

    const scroll = await page.evaluate(() => ({
      x: window.scrollX,
      y: window.scrollY,
      sw: document.documentElement.scrollWidth,
      sh: document.documentElement.scrollHeight,
      cw: document.documentElement.clientWidth,
      ch: document.documentElement.clientHeight,
    }));
    assert.equal(scroll.x, 0);
    assert.equal(scroll.y, 0);
    assert.ok(scroll.sw <= scroll.cw + 1, `horizontal overflow ${scroll.sw} > ${scroll.cw}`);
    assert.ok(scroll.sh <= scroll.ch + 1, `vertical overflow ${scroll.sh} > ${scroll.ch}`);

    const ink = await page.evaluate(() => {
      const canvas = document.querySelector('#field-canvas');
      const ctx = canvas.getContext('2d');
      const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
      let min = 255 * 3;
      let max = 0;
      for (let i = 0; i < data.length; i += 16) {
        const y = data[i] + data[i + 1] + data[i + 2];
        if (y < min) min = y;
        if (y > max) max = y;
      }
      return width > 10 && height > 10 && max - min > 8;
    });
    assert.equal(ink, true);

    if (!existsSync(join(root, 'src/fdtd.js'))) {
      assert.match(await page.locator('#field-state').innerText(), /coming soon/);
      assert.match(await page.locator('#channel-state').innerText(), /coming soon/);
    }
    if (!existsSync(join(root, 'src/signal.js')) || !existsSync(join(root, 'src/rng.js'))) {
      assert.match(await page.locator('#signal-state').innerText(), /module not ready/);
    }

    await page.click('#tool-metal');
    assert.equal(await page.getAttribute('#tool-metal', 'aria-pressed'), 'true');
    const drag = async (x0, y0, x1, y1) => {
      const live = await page.locator('#field-canvas').boundingBox();
      await page.mouse.move(live.x + x0 * live.width, live.y + y0 * live.height);
      await page.mouse.down();
      await page.mouse.move(live.x + x1 * live.width, live.y + y1 * live.height, { steps: 6 });
      await page.mouse.up();
    };
    await drag(0.12, 0.15, 0.22, 0.7);
    await page.click('#tool-dielectric');
    await drag(0.4, 0.2, 0.62, 0.45);
    await page.click('#tool-tx');
    {
      const live = await page.locator('#field-canvas').boundingBox();
      await page.mouse.click(live.x + 0.3 * live.width, live.y + 0.55 * live.height);
    }
    await page.click('#tool-rx');
    {
      const live = await page.locator('#field-canvas').boundingBox();
      const spots = [[0.5, 0.3], [0.58, 0.48], [0.7, 0.62], [0.8, 0.4], [0.86, 0.7]];
      for (const [x, y] of spots) {
        await page.mouse.click(live.x + x * live.width, live.y + y * live.height);
      }
    }
    assert.ok(Number(await page.getAttribute('#field-canvas', 'data-materials')) >= 2);
    assert.equal(await page.getAttribute('#field-canvas', 'data-source'), '1');
    assert.equal(await page.getAttribute('#field-canvas', 'data-probes'), '4');
    assert.match(await page.locator('#field-readout').innerText(), /4 receiver probes maximum/);
    await page.click('#mode-scalar');
    assert.equal(await page.getAttribute('#panel-field', 'data-mode'), 'scalar');
    await page.click('#mode-tmz');
    assert.equal(await page.getAttribute('#panel-field', 'data-mode'), 'tmz');

    if (existsSync(join(root, 'src/signal.js')) && existsSync(join(root, 'src/rng.js'))) {
      await page.click('#run-link');
      await page.waitForFunction(() => {
        const text = document.querySelector('#signal-note')?.textContent || '';
        return text.length > 0 && !text.startsWith('running');
      }, null, { timeout: 15000 });
      const note = await page.locator('#signal-note').innerText();
      assert.match(note, /offset/);
      assert.match(note, /pRef/);
    }

    assert.deepEqual(failed, [], failed.join('\n'));
    assert.deepEqual(errors, [], errors.join('\n'));

    if (process.env.WAP_SCREENSHOT) {
      await page.screenshot({ path: process.env.WAP_SCREENSHOT, fullPage: false });
    }
  } finally {
    await browser.close();
    await hosted.close();
  }
});
