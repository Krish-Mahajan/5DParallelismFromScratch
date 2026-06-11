Run a command on the RunPod pod via SSH. Check memory for connection details (runpod-sync).

Usage: /runpod <command to run on remote>

Execute the user's command on the remote pod:
```
ssh -T -i ~/.runpod/ssh/runpodctl-ssh-key -o StrictHostKeyChecking=no -p 17532 root@103.207.149.138 "<user's command>"
```

If the connection fails, the pod IP/port may have changed. Run `runpodctl ssh info em7j3z5rdwpj9m` to get the updated connection details.

Report the output back to the user.
