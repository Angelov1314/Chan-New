Chan Theory Web App – iOS App Store Deployment
=================================================

This folder contains guidance and scaffolding to package the existing Flask web app as a native iOS app using a WKWebView wrapper (Capacitor). The iOS app will load your running Flask site via HTTPS. This is the fastest, compliant path to ship on the App Store without rewriting your backend.

Overview
--------
Two supported modes:
1) Remote Web Mode (recommended): iOS app is a thin wrapper that loads your deployed Flask site at an HTTPS URL.
2) Hybrid/Local Mode (advanced): bundle static web assets locally and use API calls to a deployed backend. Since this project renders server-side, Remote Web Mode is more suitable.

What you’ll set up
------------------
- A Capacitor iOS wrapper project
- App identifiers, icons, privacy and ATS settings
- (Optional) Fastlane for automated builds and uploads

Prerequisites
-------------
- macOS with Xcode installed
- Node.js 18+
- An Apple Developer account
- Your Flask app deployed at an HTTPS URL (e.g. https://your-domain.com)

Step-by-step
------------
1) Create wrapper project (outside this repo or in ./ios_deploy/capacitor_app)
   - In Terminal:
     ```bash
     mkdir -p capacitor_app && cd capacitor_app
     npm init -y
     npm install @capacitor/core @capacitor/cli @capacitor/ios
     npx cap init "Chan Analyzer" com.yourcompany.chananalyzer --web-dir=www
     ```
   - We’ll run in Remote Web Mode, so we won’t use local www. Configure server.url next.

2) Configure Capacitor to load your Flask site
   - Create/modify capacitor.config.ts:
     ```ts
     import { CapacitorConfig } from '@capacitor/cli';

     const config: CapacitorConfig = {
       appId: 'com.yourcompany.chananalyzer',
       appName: 'Chan Analyzer',
       webDir: 'www',
       server: {
         url: 'https://your-domain.com', // <- point to your deployed Flask URL
         cleartext: false,               // enforce HTTPS
         androidScheme: 'https'
       }
     };

     export default config;
     ```

3) Add iOS platform and open Xcode
   ```bash
   npx cap add ios
   npx cap open ios
   ```

4) iOS settings in Xcode
   - Update Bundle Identifier to match your team (e.g. com.yourcompany.chananalyzer)
   - Signing & Capabilities: select your team
   - Info.plist:
     - App Transport Security: Allow Arbitrary Loads = NO (keep HTTPS). If you need subdomains, add NSExceptionDomains entries for your domains.
     - Privacy: add usage descriptions if you later access camera/photos/etc. (not required for plain web).
   - Icons/Splash: Provide required sizes (see ./assets/ for template names). You can drag in your icon set.

5) App Review compliance notes
   - The app must provide app-like value (not just a website wrapper). Highlight native-feeling UI, performance, and utility.
   - Ensure all content is accessible over HTTPS.
   - Include an in-app “Support” or “Privacy Policy” link (can be a web page).

6) Build & run on device/simulator
   - In Xcode: Product → Build / Run.
   - Verify the app loads your Flask site and navigations work.

7) Prepare for App Store Connect
   - Versioning: set CFBundleShortVersionString (1.0.0) and CFBundleVersion (build number)
   - Archive in Xcode (Product → Archive), then Distribute to App Store Connect or use Fastlane (below).

Fastlane (optional)
-------------------
We include a starter fastlane structure here. Copy the files in ./fastlane into your iOS project’s ios/App/ directory (where the Xcode project is). Then:

```bash
cd capacitor_app/ios/App
bundle init # if needed, then add fastlane to Gemfile
bundle add fastlane
bundle exec fastlane init
```

Use ./fastlane/Appfile and ./fastlane/Fastfile templates from this folder to wire your bundle id and Apple ID. Run:

```bash
bundle exec fastlane build
bundle exec fastlane upload
```

Assets
------
- Place store icons/screenshots under ./assets/ and then add to the Xcode asset catalog.

Tips & Troubleshooting
----------------------
- If your site uses cookies/sessions, ensure SameSite and Secure flags are set correctly for iOS WebView.
- If some external resources are blocked, confirm they are served via HTTPS and allowed by ATS.
- For best review outcomes, add a basic native menu (WKWebView pull-to-refresh, a native settings screen, etc.).

Repository pointers
-------------------
- Your Flask entry is at web/app.py. Deploy it to an HTTPS server (e.g., Gunicorn + Nginx) and point Capacitor server.url to that hostname.

License & Ownership
-------------------
This deployment guidance is boilerplate; adjust bundle ids, names, and app metadata to your organization.


