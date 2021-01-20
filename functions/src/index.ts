import * as cors from 'cors';
import { Key, Helpers } from 'cryptolens';
import * as functions from 'firebase-functions';

const cryptolens = functions.config().cryptolens;

exports.activate = functions.https.onRequest(async (req, res) => {
  if (!req.body.machine_code || !req.body.license_code) {
    res.status(400).send('Bad request');
    return;
  }

  let license;
  try {
    license = await Key.Activate(cryptolens.token, cryptolens.key, cryptolens.product, req.body.license_code, req.body.machine_code);
  } catch(err) {
    res.status(500).send('Internal server error')
    return;
  }

  if (!license) {
    res.status(400).send('Invalid license code');
    return;
  }

  return cors({ origin: true })(req, res, () => {
    res.status(200).json({
      license: Helpers.SaveAsString(license),
      public_key: cryptolens.key
    });
  });
});
