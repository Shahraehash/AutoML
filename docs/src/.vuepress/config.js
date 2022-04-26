const { description } = require('../../package')
const { version } = require('../../../package')

module.exports = {
  base: '/docs/',

  /**
   * Ref：https://v1.vuepress.vuejs.org/config/#title
   */
  title: `Machine Intelligence Learning Optimizer (MILO-ML) Documentation (v${version})`,
  /**
   * Ref：https://v1.vuepress.vuejs.org/config/#description
   */
  description: description,

  dest: '../static/docs',

  /**
   * Extra tags to be injected to the page HTML `<head>`
   *
   * ref：https://v1.vuepress.vuejs.org/config/#head
   */
  head: [
    ['meta', { name: 'theme-color', content: '#3880ff' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'black' }]
  ],

  /**
   * Theme configuration, here is the default theme configuration for VuePress.
   *
   * ref：https://v1.vuepress.vuejs.org/theme/default-theme-config.html
   */
  themeConfig: {
    repo: '',
    editLinks: false,
    docsDir: '',
    editLinkText: '',
    lastUpdated: false,
    nav: [
      {
        text: 'Auto-ML Guide',
        link: '/auto-ml-guide/',
      },
      {
        text: 'Processor Guide',
        link: '/processor-guide/',
      },
      {
        text: 'Install Guide',
        link: '/install-guide/'
      }
    ],
    sidebar: [
      {
        title: 'Install Guide',
        path: '/install-guide/',
        collapsable: false,
        children: [
          '/install-guide/',
          '/install-guide/docker',
          '/install-guide/firewall'
        ]
      },
      {
        title: 'Processor Guide',
        path: '/processor-guide/',
        collapsable: false,
        children: [
          '/processor-guide/',
          '/processor-guide/train-test-builder',
          '/processor-guide/general',
          '/processor-guide/multicollinearity',
          '/processor-guide/feature-selector',
          '/processor-guide/column-reducer',
        ]
      },
      {
        title: 'Auto-ML Guide',
        path: '/auto-ml-guide/',
        collapsable: false,
        children: [
          '/auto-ml-guide/',
          '/auto-ml-guide/get-started',
          '/auto-ml-guide/dataset-preparation',
          '/auto-ml-guide/homepage',
          '/auto-ml-guide/selecting-dataset',
          '/auto-ml-guide/analyzing-dataset',
          '/auto-ml-guide/model-building',
          '/auto-ml-guide/run-status',
          '/auto-ml-guide/model-results',
          '/auto-ml-guide/test-model',
          '/auto-ml-guide/publish-model',
          '/auto-ml-guide/conclusion',
          '/auto-ml-guide/glossary-terms',
          '/auto-ml-guide/glossary-report-export',
          '/auto-ml-guide/glossary-performance-export',
          '/auto-ml-guide/sample-datasets',
          '/auto-ml-guide/acknowledgments'
        ]
      }
    ]
  },

  /**
   * Apply plugins，ref：https://v1.vuepress.vuejs.org/zh/plugin/
   */
  plugins: [
    '@vuepress/plugin-back-to-top',
    '@vuepress/plugin-medium-zoom',
    '@snowdog/vuepress-plugin-pdf-export',
  ]
}
