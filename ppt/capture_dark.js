const { chromium } = require("playwright");
const path = require("path");

const BASE = "http://127.0.0.1:5180";
const OUT = path.resolve("ppt/shots");

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 1.5,
    colorScheme: "dark",
  });
  const page = await ctx.newPage();
  const click = async (re) => {
    const b = page.getByRole("button", { name: re }).first();
    if (await b.count()) await b.click();
  };
  const shot = async (name) => {
    await page.screenshot({ path: path.join(OUT, name) });
    console.log("shot", name);
  };

  await page.goto(BASE + "/dashboard", { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".station-controls", { timeout: 45000 });

  await click(/^Demo$/);
  await page.waitForTimeout(1000);
  await click(/^Dark$/);
  await click(/^Septic Shock$/);
  const start = page.getByRole("button", { name: /Start stream/i });
  if (await start.count()) await start.click();

  for (let i = 0; i < 40; i++) {
    await page.waitForTimeout(1000);
    if ((await page.locator(".bed-tile.tone-critical").count()) > 0) break;
  }
  await page.waitForTimeout(800);
  await shot("dashboard-critical-dark.png");

  const tile = page.locator(".bed-tile.tone-critical").first();
  await tile.click();
  await page.waitForTimeout(3200);
  await page.mouse.wheel(0, 950);
  await page.waitForTimeout(1200);
  await shot("bed-profile-dark-alert.png");

  console.log("url:", page.url());
  await browser.close();
})();
