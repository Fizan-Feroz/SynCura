const { chromium } = require("playwright");
const path = require("path");
const fs = require("fs");

const BASE = "http://127.0.0.1:5180";
const OUT = path.resolve("ppt/shots");
fs.mkdirSync(OUT, { recursive: true });

async function main() {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 1.5,
  });
  const page = await ctx.newPage();
  const shot = async (name) => {
    await page.screenshot({ path: path.join(OUT, name) });
    console.log("shot", name);
  };
  const click = async (re) => {
    const b = page.getByRole("button", { name: re }).first();
    if (await b.count()) await b.click();
  };

  // 1. landing
  await page.goto(BASE + "/", { waitUntil: "networkidle" });
  await page.waitForTimeout(2500);
  await shot("landing.png");

  // 2. shared backend engine
  await page.goto(BASE + "/dashboard", { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".station-controls", { timeout: 45000 });
  await click(/^Backend live$/);
  await page.waitForTimeout(3500);
  await click(/^Septic Shock$/);
  const start = page.getByRole("button", { name: /Start stream/i });
  if (await start.count()) await start.click();
  await page.waitForTimeout(6000);
  await shot("dashboard-live.png");

  // 3. demo, drive to a critical bed
  await click(/^Demo$/);
  await page.waitForTimeout(1200);
  await click(/^Septic Shock$/);
  const start2 = page.getByRole("button", { name: /Start stream/i });
  if (await start2.count()) await start2.click();

  for (let i = 0; i < 40; i++) {
    await page.waitForTimeout(1000);
    if ((await page.locator(".bed-tile.tone-critical").count()) > 0) break;
  }
  await page.waitForTimeout(800);
  await shot("dashboard-critical.png");

  await click(/^Dark$/);
  await page.waitForTimeout(900);
  await shot("dashboard-critical-dark.png");

  // 4. critical bed profile
  const tile = page.locator(".bed-tile.tone-critical").first();
  if (await tile.count()) {
    await tile.click();
    await page.waitForTimeout(3500);
    await shot("bed-profile-dark.png");
    await click(/^Light$/);
    await page.waitForTimeout(900);
    await shot("bed-profile.png");
    await page.mouse.wheel(0, 900);
    await page.waitForTimeout(1000);
    await shot("bed-profile-alert.png");
  }

  console.log("url:", page.url());
  await browser.close();
}

main().catch((e) => {
  console.error("FAILED", e);
  process.exit(1);
});
