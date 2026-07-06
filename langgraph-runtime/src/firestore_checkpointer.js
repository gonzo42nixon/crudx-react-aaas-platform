"use strict";

const { Firestore, FieldValue } = require("@google-cloud/firestore");
const {
  BaseCheckpointSaver,
  WRITES_IDX_MAP,
  TASKS,
  copyCheckpoint,
  getCheckpointId,
  maxChannelVersion
} = require("@langchain/langgraph-checkpoint");

const DEFAULT_CHECKPOINT_COLLECTION = "langgraph_checkpoints";
const DEFAULT_WRITES_COLLECTION = "langgraph_checkpoint_writes";

function encodeDocId(parts) {
  return Buffer.from(JSON.stringify(parts), "utf8").toString("base64url");
}

function requireStorageKey(field, value, { allowEmpty = false } = {}) {
  if (typeof value !== "string") {
    throw new Error(`Invalid checkpoint ${field}: expected string`);
  }
  if (!allowEmpty && value === "") {
    throw new Error(`Invalid checkpoint ${field}: empty string is not allowed`);
  }
}

function toStoredBytes(value) {
  return Buffer.from(value);
}

async function serializeJson(serde, value) {
  const [type, bytes] = await serde.dumpsTyped(value);
  return { type, bytes: toStoredBytes(bytes) };
}

async function deserializeJson(serde, stored) {
  if (!stored) return undefined;
  return serde.loadsTyped(stored.type || "json", stored.bytes);
}

class FirestoreCheckpointSaver extends BaseCheckpointSaver {
  constructor(options = {}) {
    super(options.serde);
    this.firestore = options.firestore || new Firestore();
    this.checkpointCollectionName = options.checkpointCollection || DEFAULT_CHECKPOINT_COLLECTION;
    this.writesCollectionName = options.writesCollection || DEFAULT_WRITES_COLLECTION;
  }

  checkpointCollection() {
    return this.firestore.collection(this.checkpointCollectionName);
  }

  writesCollection() {
    return this.firestore.collection(this.writesCollectionName);
  }

  checkpointDoc(threadId, checkpointNs, checkpointId) {
    return this.checkpointCollection().doc(encodeDocId([threadId, checkpointNs, checkpointId]));
  }

  writeDoc(threadId, checkpointNs, checkpointId, taskId, idx) {
    return this.writesCollection().doc(encodeDocId([threadId, checkpointNs, checkpointId, taskId, idx]));
  }

  async getTuple(config) {
    const threadId = config.configurable?.thread_id;
    const checkpointNs = config.configurable?.checkpoint_ns ?? "";
    let checkpointId = getCheckpointId(config);
    requireStorageKey("thread_id", threadId);
    requireStorageKey("checkpoint_ns", checkpointNs, { allowEmpty: true });

    let document;
    if (checkpointId) {
      requireStorageKey("checkpoint_id", checkpointId);
      const snapshot = await this.checkpointDoc(threadId, checkpointNs, checkpointId).get();
      if (!snapshot.exists) return undefined;
      document = snapshot.data();
    } else {
      document = await this.latestCheckpointDocument(threadId, checkpointNs);
      checkpointId = document?.checkpoint_id;
    }

    if (!document) return undefined;
    return this.toCheckpointTuple(document);
  }

  async latestCheckpointDocument(threadId, checkpointNs) {
    const snapshot = await this.checkpointCollection()
      .where("thread_id", "==", threadId)
      .get();
    const documents = snapshot.docs
      .map((doc) => doc.data())
      .filter((doc) => doc.checkpoint_ns === checkpointNs)
      .sort((left, right) => String(right.checkpoint_id).localeCompare(String(left.checkpoint_id)));
    return documents[0];
  }

  async *list(config, options = {}) {
    const threadId = config.configurable?.thread_id;
    const checkpointNs = config.configurable?.checkpoint_ns;
    const checkpointId = config.configurable?.checkpoint_id;
    const beforeCheckpointId = options.before?.configurable?.checkpoint_id;
    const filter = options.filter;
    let remaining = options.limit;

    let query = this.checkpointCollection();
    if (threadId) {
      requireStorageKey("thread_id", threadId);
      query = query.where("thread_id", "==", threadId);
    }

    const snapshot = await query.get();
    const documents = snapshot.docs
      .map((doc) => doc.data())
      .filter((doc) => checkpointNs === undefined || doc.checkpoint_ns === checkpointNs)
      .filter((doc) => !checkpointId || doc.checkpoint_id === checkpointId)
      .filter((doc) => !beforeCheckpointId || String(doc.checkpoint_id) < String(beforeCheckpointId))
      .sort((left, right) => String(right.checkpoint_id).localeCompare(String(left.checkpoint_id)));

    for (const document of documents) {
      if (remaining !== undefined) {
        if (remaining <= 0) break;
        remaining -= 1;
      }
      const tuple = await this.toCheckpointTuple(document);
      if (filter && !Object.entries(filter).every(([key, value]) => tuple.metadata?.[key] === value)) {
        continue;
      }
      yield tuple;
    }
  }

  async put(config, checkpoint, metadata) {
    const preparedCheckpoint = copyCheckpoint(checkpoint);
    const threadId = config.configurable?.thread_id;
    const checkpointNs = config.configurable?.checkpoint_ns ?? "";
    const parentCheckpointId = config.configurable?.checkpoint_id || null;
    requireStorageKey("thread_id", threadId);
    requireStorageKey("checkpoint_ns", checkpointNs, { allowEmpty: true });
    requireStorageKey("checkpoint_id", checkpoint.id);

    const [serializedCheckpoint, serializedMetadata] = await Promise.all([
      serializeJson(this.serde, preparedCheckpoint),
      serializeJson(this.serde, metadata)
    ]);

    await this.checkpointDoc(threadId, checkpointNs, checkpoint.id).set({
      thread_id: threadId,
      checkpoint_ns: checkpointNs,
      checkpoint_id: checkpoint.id,
      parent_checkpoint_id: parentCheckpointId,
      checkpoint: serializedCheckpoint,
      metadata: serializedMetadata,
      updated_at: FieldValue.serverTimestamp()
    }, { merge: true });

    return {
      configurable: {
        thread_id: threadId,
        checkpoint_ns: checkpointNs,
        checkpoint_id: checkpoint.id
      }
    };
  }

  async putWrites(config, writes, taskId) {
    const threadId = config.configurable?.thread_id;
    const checkpointNs = config.configurable?.checkpoint_ns ?? "";
    const checkpointId = config.configurable?.checkpoint_id;
    requireStorageKey("thread_id", threadId);
    requireStorageKey("checkpoint_ns", checkpointNs, { allowEmpty: true });
    requireStorageKey("checkpoint_id", checkpointId);
    requireStorageKey("task_id", taskId);

    await Promise.all(writes.map(async ([channel, value], idx) => {
      const writeIdx = WRITES_IDX_MAP[channel] ?? idx;
      const docRef = this.writeDoc(threadId, checkpointNs, checkpointId, taskId, writeIdx);
      if (writeIdx >= 0) {
        const existing = await docRef.get();
        if (existing.exists) return;
      }
      const serializedValue = await serializeJson(this.serde, value);
      await docRef.set({
        thread_id: threadId,
        checkpoint_ns: checkpointNs,
        checkpoint_id: checkpointId,
        task_id: taskId,
        idx: writeIdx,
        channel,
        value: serializedValue,
        updated_at: FieldValue.serverTimestamp()
      }, { merge: true });
    }));
  }

  async deleteThread(threadId) {
    requireStorageKey("thread_id", threadId);
    await this.deleteQuery(this.checkpointCollection().where("thread_id", "==", threadId));
    await this.deleteQuery(this.writesCollection().where("thread_id", "==", threadId));
  }

  async deleteQuery(query) {
    while (true) {
      const snapshot = await query.limit(450).get();
      if (snapshot.empty) return;
      const batch = this.firestore.batch();
      snapshot.docs.forEach((doc) => batch.delete(doc.ref));
      await batch.commit();
    }
  }

  async toCheckpointTuple(document) {
    const checkpoint = await deserializeJson(this.serde, document.checkpoint);
    const metadata = await deserializeJson(this.serde, document.metadata);
    const parentCheckpointId = document.parent_checkpoint_id || undefined;
    if (checkpoint?.v < 4 && parentCheckpointId) {
      await this.migratePendingSends(checkpoint, document.thread_id, document.checkpoint_ns, parentCheckpointId);
    }

    const pendingWrites = await this.pendingWrites(document.thread_id, document.checkpoint_ns, document.checkpoint_id);
    const tuple = {
      config: {
        configurable: {
          thread_id: document.thread_id,
          checkpoint_ns: document.checkpoint_ns,
          checkpoint_id: document.checkpoint_id
        }
      },
      checkpoint,
      metadata,
      pendingWrites
    };

    if (parentCheckpointId) {
      tuple.parentConfig = {
        configurable: {
          thread_id: document.thread_id,
          checkpoint_ns: document.checkpoint_ns,
          checkpoint_id: parentCheckpointId
        }
      };
    }
    return tuple;
  }

  async pendingWrites(threadId, checkpointNs, checkpointId) {
    const snapshot = await this.writesCollection()
      .where("thread_id", "==", threadId)
      .get();
    const documents = snapshot.docs
      .map((doc) => doc.data())
      .filter((doc) => doc.checkpoint_ns === checkpointNs && doc.checkpoint_id === checkpointId)
      .sort((left, right) => {
        const taskOrder = String(left.task_id).localeCompare(String(right.task_id));
        return taskOrder || Number(left.idx || 0) - Number(right.idx || 0);
      });

    return Promise.all(documents.map(async (doc) => [
      doc.task_id,
      doc.channel,
      await deserializeJson(this.serde, doc.value)
    ]));
  }

  async migratePendingSends(mutableCheckpoint, threadId, checkpointNs, parentCheckpointId) {
    const pendingSends = (await this.pendingWrites(threadId, checkpointNs, parentCheckpointId))
      .filter(([_taskId, channel]) => channel === TASKS)
      .map(([_taskId, _channel, value]) => value);
    mutableCheckpoint.channel_values ??= {};
    mutableCheckpoint.channel_values[TASKS] = pendingSends;
    mutableCheckpoint.channel_versions ??= {};
    mutableCheckpoint.channel_versions[TASKS] = Object.keys(mutableCheckpoint.channel_versions).length > 0
      ? maxChannelVersion(...Object.values(mutableCheckpoint.channel_versions))
      : this.getNextVersion(undefined);
  }
}

function createFirestoreCheckpointer() {
  return new FirestoreCheckpointSaver({
    checkpointCollection: process.env.LANGGRAPH_CHECKPOINT_COLLECTION || DEFAULT_CHECKPOINT_COLLECTION,
    writesCollection: process.env.LANGGRAPH_CHECKPOINT_WRITES_COLLECTION || DEFAULT_WRITES_COLLECTION
  });
}

module.exports = {
  FirestoreCheckpointSaver,
  createFirestoreCheckpointer
};
