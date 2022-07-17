import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AuthGuard, redirectLoggedInTo } from '@angular/fire/auth-guard';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { Routes, RouterModule } from '@angular/router';
import { IonicModule } from '@ionic/angular';

import { LoginPageComponent } from './login.page';
import { environment } from '../../../environments/environment';

const redirectAuthorizedToHome = () => redirectLoggedInTo(['/']);

const routes: Routes = [
  {
    path: 'sign-in',
    ...(environment.localUser === 'true' ? {} : { canActivate: [AuthGuard] }),
    data: { authGuardPipe: redirectAuthorizedToHome },
    component: LoginPageComponent
  },
  {
    path: 'sign-up',
    ...(environment.localUser === 'true' ? {} : { canActivate: [AuthGuard] }),
    data: { authGuardPipe: redirectAuthorizedToHome },
    component: LoginPageComponent
  },
  {
    path: 'sign-out',
    ...(environment.localUser === 'true' ? {} : { canActivate: [AuthGuard] }),
    data: { authGuardPipe: redirectAuthorizedToHome },
    component: LoginPageComponent
  },
  {
    path: 'forgot-password',
    ...(environment.localUser === 'true' ? {} : { canActivate: [AuthGuard] }),
    data: { authGuardPipe: redirectAuthorizedToHome },
    component: LoginPageComponent
  },
  {
    path: 'continue',
    component: LoginPageComponent
  },
  {
    path: 'check-email',
    ...(environment.localUser === 'true' ? {} : { canActivate: [AuthGuard] }),
    data: { authGuardPipe: redirectAuthorizedToHome },
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
