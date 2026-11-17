// Strip TypeScript types from a test file by deleting ranges, so the layout is kept.
// usage: node strip.cjs <typescript-module-dir> in.ts out.js
const path = require('path');
const fs = require('fs');
const ts = require(path.join(process.argv[2], 'typescript'));
const text = fs.readFileSync(process.argv[3], 'utf8').replace(/\r\n/g, '\n');
const sf = ts.createSourceFile('x.ts', text, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
const cuts = [];
const cut = (a, b) => cuts.push([a, b]);

function colonBefore(node) {
  // start of the ':' token that precedes node.type
  let i = node.type.getFullStart();
  while (i > 0 && text[i - 1] !== ':') i--;
  return i - 1;
}

function visit(node) {
  switch (node.kind) {
    case ts.SyntaxKind.InterfaceDeclaration:
    case ts.SyntaxKind.TypeAliasDeclaration:
      cut(node.getFullStart(), node.getEnd());
      return;
    case ts.SyntaxKind.ImportDeclaration: {
      const clause = node.importClause;
      if (clause && clause.namedBindings && ts.isNamedImports(clause.namedBindings)) {
        const els = clause.namedBindings.elements;
        const keep = els.filter(e => !e.isTypeOnly && !(clause.isTypeOnly));
        if (keep.length === 0 && !clause.name) { cut(node.getFullStart(), node.getEnd()); return; }
        if (keep.length !== els.length) {
          // rebuild the specifier list in place, keeping the original layout of kept names
          els.forEach((e, i) => {
            if (!e.isTypeOnly) return;
            let start = e.getFullStart();
            let end = e.getEnd();
            // eat the trailing comma and whitespace up to the newline
            let j = end;
            if (text[j] === ',') j++;
            while (text[j] === ' ') j++;
            if (text[j] === '\n') j++;
            // include the leading newline+indent
            let k = e.getStart();
            while (k > 0 && (text[k - 1] === ' ')) k--;
            cut(k, j);
          });
        }
      }
      break;
    }
    case ts.SyntaxKind.AsExpression:
    case ts.SyntaxKind.SatisfiesExpression:
      cut(node.expression.getEnd(), node.getEnd());
      break;
    case ts.SyntaxKind.NonNullExpression:
      cut(node.getEnd() - 1, node.getEnd());
      break;
    case ts.SyntaxKind.ArrowFunction:
      if (node.parameters.length === 1 && node.parameters[0].type && !node.typeParameters &&
          !node.parameters[0].initializer && !node.parameters[0].dotDotDotToken && !node.type) {
        const prm = node.parameters[0];
        const open = text.lastIndexOf('(', prm.getStart());
        let close = prm.type.getEnd();
        while (text[close] !== ')') close++;
        cut(open, open + 1);
        cut(close, close + 1);
      }
      break;
    case ts.SyntaxKind.Parameter:
      if (node.questionToken) cut(node.questionToken.getStart(), node.questionToken.getEnd());
      break;
    case ts.SyntaxKind.CallExpression:
    case ts.SyntaxKind.NewExpression:
      if (node.typeArguments && node.typeArguments.length) {
        const a = node.expression.getEnd();
        cut(a, node.typeArguments.end + 1);
      }
      break;
    default:
      break;
  }
  if (node.type && !ts.isTypeAliasDeclaration(node) && node.kind !== ts.SyntaxKind.AsExpression &&
      node.kind !== ts.SyntaxKind.TypeAssertionExpression && node.kind !== ts.SyntaxKind.SatisfiesExpression &&
      node.kind !== ts.SyntaxKind.TypePredicate && !ts.isTypeNode(node)) {
    cut(colonBefore(node), node.type.getEnd());
  }
  if (node.typeParameters && node.typeParameters.length && !ts.isInterfaceDeclaration(node)) {
    cut(node.typeParameters.pos - 1, node.typeParameters.end + 1);
  }
  ts.forEachChild(node, visit);
}
visit(sf);

cuts.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
const accepted = [];
let curEnd = -1;
for (const c of cuts) {
  if (c[0] < curEnd) continue; // nested inside an accepted range
  accepted.push(c);
  curEnd = c[1];
}
let out = text;
for (const [a, b] of accepted.reverse()) out = out.slice(0, a) + out.slice(b);
out = out.replace(/[ \t]+$/gm, '').replace(/\n{3,}/g, '\n\n');
fs.writeFileSync(process.argv[4], out.replace(/\n/g, '\r\n'), 'utf8');
