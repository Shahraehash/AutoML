// This file can be replaced during build by using the `fileReplacements` array.
// `ng build --prod` replaces `environment.ts` with `environment.prod.ts`.
// The list of file replacements can be found in `angular.json`.

export const environment = {
  production: false,
  apiUrl: 'http://localhost:5000',
  firebase: {
    apiKey: 'AIzaSyC3mSDzUoZsTGsQpvRddhI_88R8UqUA6l8',
    authDomain: 'milo-ml.firebaseapp.com',
    databaseURL: 'https://milo-ml.firebaseio.com',
    projectId: 'milo-ml',
    storageBucket: 'milo-ml.appspot.com',
    messagingSenderId: '508550799654',
    appId: '1:508550799654:web:b8852c44b3c13dc0e461e7',
    measurementId: 'G-XK86C1E9Z4'
  }
};

/*
 * For easier debugging in development mode, you can import the following file
 * to ignore zone related error stack frames such as `zone.run`, `zoneDelegate.invokeTask`.
 *
 * This import should be commented out in production mode because it will have a negative impact
 * on performance if an error is thrown.
 */
// import 'zone.js/dist/zone-error';  // Included with Angular CLI.
