import { environment as defaults } from './environment.default';

export const environment = {
  ...defaults,
  name: 'docker',
  apiUrl: '.',
  localUser: '${{LOCAL_USER}}',
  ldapAuth: '${{LDAP_AUTH}}',
  production: true
};
