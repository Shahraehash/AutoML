import { environment as defaults } from './environment.default';

export const environment = {
  ...defaults,
  name: 'dev',
  apiUrl: 'http://localhost:5000'
};
