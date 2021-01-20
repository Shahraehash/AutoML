import * as cors from 'cors';
import * as functions from 'firebase-functions';
import { Key, Helpers } from 'cryptolens';

const cryptolens = functions.config().cryptolens;

exports.activate = functions.https.onRequest(async (req, res) => {
  return cors({origin: true})(req, res, () => {
    res.status(200).send(cryptolens.product);
  });
});
