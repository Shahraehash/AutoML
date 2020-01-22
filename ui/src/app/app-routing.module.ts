import { NgModule } from '@angular/core';
import { PreloadAllModules, RouterModule, Routes } from '@angular/router';
import { AngularFireAuthGuard, redirectUnauthorizedTo } from '@angular/fire/auth-guard';

import { environment } from '../environments/environment';

const redirectUnauthorizedToLogin = () => redirectUnauthorizedTo(['login', {redirectTo: location.pathname}]);

const routes: Routes = [
  { path: 'login', loadChildren: () => import('./pages/login/login.module').then(m => m.LoginPageModule) },
  {
    path: 'search',
    ...(environment.localUser ? {} : {canActivate: [AngularFireAuthGuard]}),
    data: { authGuardPipe: redirectUnauthorizedToLogin },
    loadChildren: () => import('./pages/search/search.module').then(m => m.SearchPageModule)
  },
  {
    path: 'search/:dataId/:step',
    ...(environment.localUser ? {} : {canActivate: [AngularFireAuthGuard]}),
    data: { authGuardPipe: redirectUnauthorizedToLogin },
    loadChildren: () => import('./pages/search/search.module').then(m => m.SearchPageModule)
  },
  {
    path: 'search/:dataId/job/:jobId/:step',
    ...(environment.localUser ? {} : {canActivate: [AngularFireAuthGuard]}),
    data: { authGuardPipe: redirectUnauthorizedToLogin },
    loadChildren: () => import('./pages/search/search.module').then(m => m.SearchPageModule)
  },
  {
    path: 'search/:dataId/job/:jobId/:step/:taskId/status',
    ...(environment.localUser ? {} : {canActivate: [AngularFireAuthGuard]}),
    data: { authGuardPipe: redirectUnauthorizedToLogin },
    loadChildren: () => import('./pages/search/search.module').then(m => m.SearchPageModule)
  },
  {
    path: 'model/:id',
    loadChildren: () => import('./pages/run-model/run-model.module').then(m => m.RunModelPageModule)
  },
  { path: '**', redirectTo: 'search' }
];

@NgModule({
  imports: [
    RouterModule.forRoot(routes, { preloadingStrategy: PreloadAllModules })
  ],
  exports: [RouterModule]
})
export class AppRoutingModule {}
