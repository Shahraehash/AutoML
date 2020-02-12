import { environment as defaults } from './environment.default';

export const environment = {
  ...defaults,
  apiUrl: 'https://api.miloml.com',
  production: true
};
