# CodeForge Storage Controller privacy policy

Last updated: October 26, 2026.

## Purpose and operator

CodeForge Storage Controller is a personal project-backup configuration operated by [Yash Bajpai](https://github.com/Yash1bajpai) for the [CodeForge project](https://github.com/Yash1bajpai/CodeForge-250M). It is not a public hosted storage service. The configuration runs rclone and project-control commands on the owner's phone.

## Google Drive access

The Google Drive authorization permits reading and writing Drive files, including creating backup files, listing project folders, downloading files for verification, and deleting the controller's own temporary verification files. The requested OAuth scope is `https://www.googleapis.com/auth/drive`.

This is a broad account permission, not an OAuth-enforced folder restriction. The documented project commands target the owner's `codeforge-storage` folder and operate on owner-selected project artifacts. They do not scan unrelated Drive content or delete unrelated files. Backup procedures use copy operations rather than mirror/sync deletion.

## Data handling

- Project artifacts can include source snapshots, dataset preparation outputs, model checkpoints, evaluations and logs selected by the owner.
- OAuth credentials are stored outside the source repository in private phone application storage. Credentials are not published in Git, notebook source, public logs or this policy.
- Verification can temporarily download project files to the phone and compare their byte counts and checksums. Status reports record operational metadata such as file sizes, hashes, timestamps and success/failure results, not OAuth tokens.
- Google stores Drive files and processes API requests under its own terms and privacy policy. rclone communicates directly with Google; this configuration does not run an additional public intermediary server.
- This backup client does not itself train models or send unrelated Google Drive content to an AI provider. CodeForge model training is a separate Kaggle/Hugging Face workflow using selected, licensed sources.

The operator does not sell Google user data or use it for advertising. Access and use of information received from Google APIs will adhere to the [Google API Services User Data Policy](https://developers.google.com/terms/api-services-user-data-policy), including its Limited Use requirements.

## Retention and removal

Project backups remain until the owner removes them. Local verification copies and logs remain under the owner's control; deletion of a local copy does not delete a cloud backup. Temporary verification objects are removed after their readback is verified when the cleanup operation succeeds.

The owner can revoke this client's authorization through [Google Account third-party connections](https://myaccount.google.com/connections). Revocation stops authorized API access but does not automatically delete previously stored backup files. The owner can separately delete unwanted Drive backups, local project copies and the private rclone configuration.

## Contact and changes

For privacy questions or deletion requests, contact the operator through the [project issue tracker](https://github.com/Yash1bajpai/CodeForge-250M/issues). Do not include credentials, private files or personal information in a public issue. This page will be updated if the controller's documented data handling changes.
