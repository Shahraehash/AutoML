import { environment as defaults } from './environment.default';

export const environment = {
  ...defaults,
  name: 'docker',
  apiUrl: '.',
  localUser: false,
  production: true
};
