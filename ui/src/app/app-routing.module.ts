import { NgModule } from '@angular/core';
import { PreloadAllModules, RouterModule, Routes } from '@angular/router';
import { canActivate, redirectUnauthorizedTo } from '@angular/fire/auth-guard';

const routes: Routes = [
  { path: 'login', loadChildren: () => import('./pages/login/login.module').then(m => m.LoginPageModule) },
  {
    path: 'search',
    ...canActivate(redirectUnauthorizedTo(['login'])),
    loadChildren: () => import('./pages/search/search.module').then(m => m.SearchPageModule)
  },
  {
    path: 'search/:dataId/:step',
    ...canActivate(redirectUnauthorizedTo(['login'])),
    loadChildren: () => import('./pages/search/search.module').then(m => m.SearchPageModule)
  },
  {
    path: 'search/:dataId/job/:jobId/:step',
    ...canActivate(redirectUnauthorizedTo(['login'])),
    loadChildren: () => import('./pages/search/search.module').then(m => m.SearchPageModule)
  },
  {
    path: 'search/:dataId/job/:jobId/:step/:taskId/status',
    ...canActivate(redirectUnauthorizedTo(['login'])),
    loadChildren: () => import('./pages/search/search.module').then(m => m.SearchPageModule)
  },
  {
    path: 'model/:id',
    ...canActivate(redirectUnauthorizedTo(['login'])),
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
