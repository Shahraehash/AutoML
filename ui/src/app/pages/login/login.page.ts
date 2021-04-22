import { Component } from '@angular/core';
import { AngularFireAuth } from '@angular/fire/auth';
import { Router, ActivatedRoute } from '@angular/router';
import firebase from 'firebase/app';
import 'firebase/auth';

@Component({
  selector: 'app-login',
  templateUrl: './login.page.html',
  styleUrls: ['./login.page.scss'],
})
export class LoginPageComponent {

  constructor(
    private route: ActivatedRoute,
    private afAuth: AngularFireAuth,
    private router: Router
  ) {}

  async login() {
    const redirect = this.route.snapshot.params.redirectTo;
    await this.afAuth.signInWithPopup(new firebase.auth.GoogleAuthProvider());
    this.router.navigateByUrl(redirect ? redirect : '/search');
  }
}
