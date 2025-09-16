import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.chananalysis.app',
  appName: 'Chan Analysis',
  webDir: 'dist',
  server: {
    url: 'https://chan-new.onrender.com',
    cleartext: false
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 2000,
      backgroundColor: "#ffffff",
      showSpinner: true,
      spinnerColor: "#999999"
    }
  }
};

export default config;
