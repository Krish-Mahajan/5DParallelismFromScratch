Sync files with the RunPod pod. Check memory for connection details (runpod-sync).

First verify the pod is still running:
```
runpodctl pod list
```

If the pod IP/port has changed, run `runpodctl ssh info em7j3z5rdwpj9m` and update `sync.sh` accordingly.

Then run the requested sync operation:
- Push local to remote: `./sync.sh push`
- Pull remote to local: `./sync.sh pull`
- Start auto-sync watcher: `./sync.sh watch` (runs in background)

If the user says "sync" without specifying direction, default to push.

After syncing, report what was transferred.
