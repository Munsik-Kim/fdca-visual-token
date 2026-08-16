# Architecture and dataflow

```mermaid
flowchart LR
 A[Reference and W8 one-step laws] --> B[Conditional coordinatewise maximal coupling]
 B --> C[Natural seed vector]
 C --> D[Common approximate suffix]
 D --> E[Descendant token mismatches]
 E --> F[Paired VQ-decoded images]
 F --> G[LPIPS and DINOv2]
 G --> H[Split incidence times conditional consequence]
```

The schedule is branch-independent and accepted Halton tokens are immutable. Event-addressed semantic keys separate reference-shared proposal draws from acceptance/residual draws and make unintended key reuse auditable.
