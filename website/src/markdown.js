import React from 'react'
import rehypeReact from 'rehype-react'

import Link, { Button } from './components/link'
import { H2, H3, H4, H5, Hr, Tag, InlineList } from './components/typography'
import { Table, Tr, Th, Td } from './components/table'
import { CodeComponent, Pre, Kbd, TypeAnnotation, Ndarray } from './components/code'
import { Infobox } from './components/box'
import Grid from './components/grid'
import { Tabs, Tab } from './components/tabs'
import Icon, { Emoji } from './components/icon'

export const renderAst = new rehypeReact({
    createElement: React.createElement,
    components: {
        h2: H2,
        h3: H3,
        h4: H4,
        h5: H5,
        hr: Hr,
        a: Link,
        button: Button,
        code: CodeComponent,
        pre: Pre,
        kbd: Kbd,
        blockquote: Infobox,
        infobox: Infobox,
        grid: Grid,
        tag: Tag,
        i: Icon,
        emoji: Emoji,
        table: Table,
        tr: Tr,
        th: Th,
        td: Td,
        tt: TypeAnnotation,
        ndarray: Ndarray,
        'inline-list': InlineList,
        tabs: Tabs,
        tab: Tab,
    },
}).Compiler
