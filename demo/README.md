# Demo evidence bundle

Upload these files together as a new case (or click **Load demo case**):

* `demo_upi_receipt.png`  - UPI payment success screen (amount, payee UPI ID)
* `demo_whatsapp_chat.png` - WhatsApp conversation with the 'inspector' (phone number, tactic)
* `demo_victim_statement.txt` - paste as the victim statement
* `demo_bank_record.json` - synthetic bank-side record (transaction time, alert log, scam-warning log, reporting times) used by the liability engine

Expected outcome: the new case shares one UPI ID with the right-hand end of hidden Cluster A. Following the chain (UPI -> phone -> UPI -> phone) recovers all 6 related reports, including a courier-parcel report that uses a different tactic. The 1930 helpline mentioned in the statement is a known legitimate number, so it must never be followed as a link.

Liability tab: the bank's app log shows a scam warning was dismissed, so customer negligence is proven (R4). The fraud was reported to the bank and to 1930 the next day, so compensation of Rs 25,000 is due (85% of Rs 48,500, capped). The transaction is from October 2026, before the framework starts, so the decision is marked as a preview. Remove `demo_bank_record.json` and the tab instead says which facts are missing.
