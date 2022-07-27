import { NgModule } from '@angular/core';
import { PreloadAllModules, RouterModule, Routes } from '@angular/router';
import { AuthGuard, redirectUnauthorizedTo } from '@angular/fire/auth-guard';

import { environment } from '../environments/environment';
import { LDAPAuthGuard } from './services';

const redirectUnauthorizedToLogin = () => redirectUnauthorizedTo(['auth/sign-in', { redirectTo: location.pathname }]);

const routes: Routes = [
  {
    path: 'auth',
    loadChildren: () => import('./pages/login/login.module').then(m => m.LoginPageModule)
  }
];

let routeMetaData;
if (environment.ldapAuth === 'true') {
  routeMetaData = {
    canActivate: [LDAPAuthGuard]
  };
} else {
  routeMetaData = {
    ...(environment.localUser === 'true' ? {} : { canActivate: [AuthGuard] }),
    data: { authGuardPipe: redirectUnauthorizedToLogin }
  };
}

if (environment.authOnly) {
  routes.push({ path: '**', redirectTo: 'auth/sign-in' });
} else {
  routes.push(
    {
      path: 'search',
      ...routeMetaData,
      loadChildren: () => import('./pages/search/search.module').then(m => m.SearchPageModule)
    },
    {
      path: 'search/:dataId/:step',
      ...routeMetaData,
      loadChildren: () => import('./pages/search/search.module').then(m => m.SearchPageModule)
    },
    {
      path: 'search/:dataId/job/:jobId/:step',
      ...routeMetaData,
      loadChildren: () => import('./pages/search/search.module').then(m => m.SearchPageModule)
    },
    {
      path: 'search/:dataId/job/:jobId/:step/:taskId/status',
      ...routeMetaData,
      loadChildren: () => import('./pages/search/search.module').then(m => m.SearchPageModule)
    },
    {
      path: 'model/:id',
      ...routeMetaData,
      loadChildren: () => import('./pages/run-model/run-model.module').then(m => m.RunModelPageModule)
    },
    {
      path: 'update-license',
      loadChildren: () => import('./pages/update-license/update-license.module').then(m => m.UpdateLicensePageModule)
    },
    {
      path: 'home',
      ...routeMetaData,
      loadChildren: () => import('./pages/home/home.module').then(m => m.HomePageModule)
    },
    { path: '**', redirectTo: environment.name === 'docker' ? 'home' : 'search' }
  );
}

@NgModule({
  imports: [
    RouterModule.forRoot(routes, { preloadingStrategy: PreloadAllModules, relativeLinkResolution: 'legacy' })
  ],
  exports: [RouterModule]
})
export class AppRoutingModule { }
