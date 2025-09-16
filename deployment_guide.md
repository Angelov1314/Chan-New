# 缠论分析系统 - 部署指南

## 方案1: PythonAnywhere (推荐) 🎯

### 步骤：
1. 访问 https://www.pythonanywhere.com
2. 注册免费账户
3. 在 "Files" 中上传项目文件
4. 在 "Web" 中创建新的 Web App
5. 选择 "Manual configuration" -> "Python 3.10"
6. 配置 WSGI 文件路径：`/home/yourusername/Chan-New/pythonanywhere_wsgi.py`
7. 安装依赖：`pip3.10 install --user -r web/requirements.txt`
8. 重启 Web App

### 获得域名：
- 免费：`yourusername.pythonanywhere.com`
- 付费：自定义域名

## 方案2: Railway

### 步骤：
1. 访问 https://railway.app
2. 连接 GitHub 仓库
3. 自动部署
4. 获得 HTTPS 域名

## 方案3: Heroku

### 步骤：
1. 访问 https://heroku.com
2. 创建新应用
3. 连接 GitHub
4. 自动部署

## iOS 应用配置

部署成功后，在 iOS 项目中：
1. 更新 `capacitor.config.ts` 中的服务器URL
2. 重新构建 iOS 应用
3. 提交到 App Store

## 当前状态
- ✅ 代码已推送到 GitHub
- ✅ 多语言系统完成
- ✅ 投资指数算法完成
- ✅ 自选股功能完成
- 🔄 等待后端部署
