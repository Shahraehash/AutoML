import { Component } from '@angular/core';
import { AngularFireAuth } from '@angular/fire/auth';
import { Router, ActivatedRoute } from '@angular/router';
import { auth } from 'firebase/app';

@Component({
  selector: 'app-login',
  templateUrl: './login.page.html',
  styleUrls: ['./login.page.scss'],
})
export class LoginPage {

  constructor(
    private route: ActivatedRoute,
    private afAuth: AngularFireAuth,
    private router: Router
  ) {}

  async login() {
    const redirect = this.route.snapshot.params.redirectTo;
    await this.afAuth.auth.signInWithPopup(new auth.GoogleAuthProvider());
    this.router.navigateByUrl(redirect ? redirect : '/search');
  }
}
