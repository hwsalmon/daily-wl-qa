Daily QA - WL - DLG - Picket Fence  (portable)

Nothing to install: unzip this folder anywhere (a local disk or a USB drive) and double-click  start-daily-qa.bat.
It carries its own Python and every library; it does not touch the rest of the computer and needs no internet.

First use
  1. Settings > Data Folders: set each machine's EPID image folder (the network share where the iView images land) and, if you want machine logs, its log folder.
  2. Check EPID Folder > Import all new sessions: analyses every Daily WL session found there (for example all of them since June) and fills the trend history. It can be cancelled and
     continued later; sessions it cannot trust are listed and left for review. Without a log folder (or with logs that do not match) sessions are analysed without logs.
  3. History / Reports: review the stored results of any day, week or month and print the report for it - a draft, or signed by the physicist (signature, date and time, comment and a digest of the
     data; the sign-off is recorded and can be re-checked later). The week and month reports show every session and the trend lines.
  4. Trends: isocenter radius and drift, jaw size, MLC size, picket fence (max, RMS, worst leaf, tilt, same-leaf test).

Your data
  The history database (wl_qa_history.db) and settings (wl_qa_config.json) are kept in the  app  folder, beside the program. To update to a newer version, unzip it to a NEW folder and copy
  those two files into its  app  folder (or unzip over the old folder: the zip contains neither file). backup-data.bat copies both to a dated folder.

Pass / fail
  Winston-Lutz: the largest displacement of any beam from the isocenter (after the static phantom offset is removed) must be 1.0 mm or less (TG-198 / MPPG, SRS). Change TOLERANCE_MM in app\wl_qa_tool.py for other limits.
