import { environment as defaults } from './environment.default';

export const environment = {
  ...defaults,
  name: 'web',
  apiUrl: 'https://api.miloml.com',
  production: true
};
