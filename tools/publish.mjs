import { spawnSync } from "node:child_process";

const commitMessage = process.argv.slice(2).join(" ").trim();

if (!commitMessage || commitMessage === "--help" || commitMessage === "-h") {
  console.log('用法：npm run publish -- "提交说明"');
  console.log("脚本会先构建网站，成功后再提交全部改动并推送到 GitHub。");
  process.exit(commitMessage ? 0 : 1);
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: process.cwd(),
    encoding: "utf8",
    stdio: options.capture ? "pipe" : "inherit",
  });

  if (result.error) {
    console.error(`无法运行 ${command}：${result.error.message}`);
    process.exit(1);
  }

  if (result.status !== 0) {
    if (options.capture && result.stderr) {
      console.error(result.stderr.trim());
    }
    process.exit(result.status ?? 1);
  }

  return options.capture ? result.stdout.trim() : "";
}

const branch = run("git", ["branch", "--show-current"], { capture: true });
if (branch !== "master") {
  console.error(`当前分支是 ${branch || "未知分支"}，请切换到 master 后再发布。`);
  process.exit(1);
}

console.log("\n1/4 构建网站");
run(process.execPath, ["node_modules/hexo/bin/hexo", "generate"]);

const changes = run("git", ["status", "--short"], { capture: true });
if (changes) {
  console.log("\n2/4 准备以下改动");
  console.log(changes);
  run("git", ["add", "--all"]);

  console.log("\n3/4 创建提交");
  run("git", ["commit", "-m", commitMessage]);
} else {
  console.log("\n2/4 没有新的文件改动");
  console.log("3/4 无需创建提交");
}

console.log("\n4/4 推送到 GitHub");
run("git", ["push", "origin", "HEAD"]);

console.log("\n发布请求已推送；GitHub Actions 将自动构建并部署网站。");
