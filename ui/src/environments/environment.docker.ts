import { environment as defaults } from './environment.default';

export const environment = {
  ...defaults,
  name: 'docker',
  apiUrl: '.',
  localUser: '${LOCAL_USER}',
  production: true
};
