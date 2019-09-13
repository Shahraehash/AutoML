import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { FormBuilder, FormGroup } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

import { parse } from 'papaparse';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage implements OnInit {
  labels = [];
  SERVER_URL = 'http://localhost:5000/upload';
  uploadForm: FormGroup;

  constructor(
    private formBuilder: FormBuilder,
    private httpClient: HttpClient,
    private router: Router
  ) {}

  ngOnInit() {
    this.uploadForm = this.formBuilder.group({
      label_column: '',
      train: [''],
      test: ['']
    });
  }

  onSubmit() {
    const formData = new FormData();
    formData.append('train', this.uploadForm.get('train').value);
    formData.append('test', this.uploadForm.get('test').value);
    formData.append('label_column', this.uploadForm.get('label_column').value);

    this.httpClient.post<any>(this.SERVER_URL, formData).subscribe(
      () => this.router.navigate(['/train', {upload: true}]),
      (err) => console.log(err)
    );

    return false;
  }

  onFileSelect(event) {
    if (event.target.files.length === 1) {
      const file = event.target.files[0];

      parse(file, {
        complete: reply => this.labels = reply.data[0]
      });

      this.uploadForm.get(event.target.name).setValue(file);
    }
  }
}
