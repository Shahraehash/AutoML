import { environment as defaults } from './environment.default';

export const environment = {
  ...defaults,
  apiUrl: '.',
  localUser: true,
  production: true
};
