# HackTheHaystack
A Git repo for the Hack the Haystack hackathon

##Running

In Submission/HackTheHaystack run:
```js
$ python3 main.py -p <path/to/email/csv/> -t <suspiciousness-threshold> -n <number-of-rows-to-import>
```

Computing suspiciousness for all emails in the file is computationally expensive (partly due to implementation), so the number of rows to load from a file can be restricted with `--nrows` or `-n`.

If a user's activity exceeds `-t` at any point, they are flagged.

`--threshold` (or `-t`) should typically be around 0.02.



