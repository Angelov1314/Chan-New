# iOS 应用部署指南

## 后端服务器
- **URL**: https://chan-new.onrender.com/
- **状态**: ✅ 已部署并运行

## iOS 应用配置

### 1. 更新服务器配置
```bash
# 在 iOS 项目目录中
npx cap sync
```

### 2. 构建 iOS 应用
```bash
# 添加 iOS 平台
npx cap add ios

# 打开 Xcode
npx cap open ios
```

### 3. 在 Xcode 中
1. 选择你的开发团队
2. 设置 Bundle Identifier: `com.chananalysis.app`
3. 配置签名证书
4. 构建并运行

### 4. 发布到 App Store
1. 在 Xcode 中选择 "Product" -> "Archive"
2. 上传到 App Store Connect
3. 提交审核

## 功能特性
- ✅ 缠论技术分析
- ✅ 多语言支持 (中文/英文)
- ✅ 股票收藏功能
- ✅ 准确性验证
- ✅ 实时数据更新

## 注意事项
- 确保网络连接正常
- 服务器可能偶尔重启，请耐心等待
- 建议在WiFi环境下使用以获得最佳体验