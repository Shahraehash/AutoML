import { Injectable } from '@angular/core';
import { Router, CanActivate, UrlTree } from '@angular/router';

import { MiloApiService } from '../milo-api/milo-api.service';

/**
 * This guard is used to check if the user is authenticated when using LDAP.
 */
@Injectable({
    providedIn: 'root'
})
export class LDAPAuthGuard implements CanActivate {
    constructor(
        private api: MiloApiService,
        private router: Router
    ) {}

    async canActivate(): Promise<boolean | UrlTree> {
       if (this.api.ldapToken) {
            return true;
        } else {
            return this.router.parseUrl(`auth/sign-in?redirectTo=${location.pathname}`);
        }
    }
}

/**
 * This guard is used to check if the user is a guest when using LDAP.
 */
@Injectable({
    providedIn: 'root'
})
export class LDAPGuestGuard implements CanActivate {
    constructor(
        private api: MiloApiService,
        private router: Router
    ) {}

    async canActivate(): Promise<boolean | UrlTree> {
       if (!this.api.ldapToken) {
            return true;
        } else {
            return this.router.parseUrl(`/`);
        }
    }
}