# Android 应用部署指南

## 后端服务器
- **URL**: https://chan-new.onrender.com/
- **状态**: ✅ 已部署并运行

## 环境准备

### 1. 安装 Android Studio
- 下载: https://developer.android.com/studio
- 安装 Android SDK
- 配置环境变量

### 2. 安装 Node.js 和 Capacitor
```bash
# 安装 Node.js (如果还没有)
# 下载: https://nodejs.org/

# 安装 Capacitor CLI
npm install -g @capacitor/cli
```

## Android 应用配置

### 1. 初始化项目
```bash
# 在项目根目录
npm init -y
npm install @capacitor/core @capacitor/cli @capacitor/android
npx cap init "Chan Analysis" "com.chananalysis.app"
```

### 2. 添加 Android 平台
```bash
npx cap add android
npx cap sync
```

### 3. 配置 Android 权限
在 `android/app/src/main/AndroidManifest.xml` 中添加网络权限：
```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

### 4. 构建和运行
```bash
# 同步代码
npx cap sync android

# 打开 Android Studio
npx cap open android

# 或者直接运行
npx cap run android
```

## 在 Android Studio 中

### 1. 配置签名
- 生成签名密钥
- 配置 `build.gradle`
- 设置签名配置

### 2. 构建 APK
- 选择 "Build" -> "Build Bundle(s) / APK(s)" -> "Build APK(s)"
- 生成 `app-debug.apk` 或 `app-release.apk`

### 3. 测试应用
- 连接 Android 设备
- 启用开发者选项和USB调试
- 运行应用进行测试

## 发布到 Google Play

### 1. 准备发布版本
- 生成签名 APK
- 测试所有功能
- 准备应用图标和截图

### 2. 上传到 Google Play Console
- 访问 https://play.google.com/console
- 创建新应用
- 上传 APK 文件
- 填写应用信息
- 提交审核

## 功能特性
- ✅ 缠论技术分析
- ✅ 多语言支持 (中文/英文)
- ✅ 股票收藏功能
- ✅ 准确性验证
- ✅ 实时数据更新
- ✅ 离线缓存支持

## 注意事项
- 确保网络连接正常
- 服务器可能偶尔重启，请耐心等待
- 建议在WiFi环境下使用以获得最佳体验
- 首次启动可能需要较长时间加载数据
