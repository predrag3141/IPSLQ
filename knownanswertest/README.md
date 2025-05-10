To run this sample, use these commands:

```bash
git clone https://github.com/predrag3141/IPSLQ.git
cd IPSLQ
git checkout toy_example
cd knownanswertest
go test -v pslqcontext.go pslqcontext_test.go
```

In a small percentage of runs, the test fails. This is most likely because the random relation seeded into the random input happens to be all zero, or is zero in all but one coordinate. If the test fails, just re-run it until a useable relation is generated.
