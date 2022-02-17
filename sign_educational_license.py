from api.licensing import request_activation

def sign_educational_license():
    """Generates an educational license key"""

    license = 'LNQYL-PHQOK-QAZFT-CRCAM'
    result = request_activation(license)

    if result is not None:
        with open('licensefile.skm', 'w') as file:
            file.write(result['license'])

        with open('license.pub', 'w') as file:
            file.write(result['public_key'])

        return print('License key generated successfully!')
    else:
        print('Unable to activate license: ' + license)

if __name__ == "__main__":
    sign_educational_license()
