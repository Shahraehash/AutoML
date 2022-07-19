import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AuthGuard, redirectLoggedInTo } from '@angular/fire/auth-guard';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { Routes, RouterModule } from '@angular/router';
import { IonicModule } from '@ionic/angular';

import { LoginPageComponent } from './login.page';
import { environment } from '../../../environments/environment';

const redirectAuthorizedToHome = () => redirectLoggedInTo(['/']);

let routeMetaData = {
  ...(environment.localUser === 'true' ? {} : { canActivate: [AuthGuard] }),
  data: { authGuardPipe: redirectAuthorizedToHome }
};

const routes: Routes = [
  {
    path: 'sign-in',
    ...routeMetaData,
    component: LoginPageComponent
  },
  {
    path: 'sign-up',
    ...routeMetaData,
    component: LoginPageComponent
  },
  {
    path: 'sign-out',
    ...routeMetaData,
    component: LoginPageComponent
  },
  {
    path: 'forgot-password',
    ...routeMetaData,
    component: LoginPageComponent
  },
  {
    path: 'continue',
    component: LoginPageComponent
  },
  {
    path: 'check-email',
    ...routeMetaData,
    component: LoginPageComponent
  },
];

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    IonicModule,
    RouterModule.forChild(routes)
  ],
  declarations: [LoginPageComponent]
})
export class LoginPageModule {}
