import { initializeApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut } from 'firebase/auth';
import { getFirestore, doc, getDoc, updateDoc, getDocFromServer } from 'firebase/firestore';
// Use glob import to safely handle missing config file at build time
const configFiles = import.meta.glob('../firebase-applet-config.json', { eager: true });
const firebaseConfig = (configFiles['../firebase-applet-config.json'] as any)?.default || {
  projectId: "gen-lang-client-0400823843",
  appId: "1:1026415005415:web:cf6d1b2f38a0c7ac1f9c42",
  apiKey: "AIzaSyBjvw8ewBhuVhjqy86LqeNNTw-egmdAXW8",
  authDomain: "gen-lang-client-0400823843.firebaseapp.com",
  firestoreDatabaseId: "ai-studio-43ca046c-b369-457f-ab19-e34c45d76090",
  storageBucket: "gen-lang-client-0400823843.firebasestorage.app",
  messagingSenderId: "1026415005415"
};

// Use environment variables if available, otherwise fallback to the JSON config
const config = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY || firebaseConfig.apiKey,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || firebaseConfig.authDomain,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || firebaseConfig.projectId,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || firebaseConfig.storageBucket,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || firebaseConfig.messagingSenderId,
  appId: import.meta.env.VITE_FIREBASE_APP_ID || firebaseConfig.appId,
  firestoreDatabaseId: import.meta.env.VITE_FIREBASE_FIRESTORE_DATABASE_ID || firebaseConfig.firestoreDatabaseId,
};

// Validate that we have real values before initializing
const isConfigValid = !!config.apiKey && !config.apiKey.includes('YOUR_API_KEY');

let app: any;
let auth: any;
let db: any;
let googleProvider: any;

try {
  if (!isConfigValid) {
    console.warn("Firebase configuration is missing or invalid. App will run in limited mode.");
    // Initialize with dummy values to prevent module load crash, but it will still fail on use
    app = initializeApp({
      apiKey: "missing",
      authDomain: "missing",
      projectId: "missing",
      storageBucket: "missing",
      messagingSenderId: "missing",
      appId: "missing"
    });
  } else {
    app = initializeApp(config);
  }
  
  auth = getAuth(app);
  db = getFirestore(app, config.firestoreDatabaseId || undefined);
  googleProvider = new GoogleAuthProvider();
} catch (e) {
  console.error("Firebase Initialization Error:", e);
}

export { auth, db, googleProvider };

export enum OperationType {
  CREATE = 'create',
  UPDATE = 'update',
  DELETE = 'delete',
  LIST = 'list',
  GET = 'get',
  WRITE = 'write',
}

export interface FirestoreErrorInfo {
  error: string;
  operationType: OperationType;
  path: string | null;
  authInfo: {
    userId: string | undefined;
    email: string | null | undefined;
    emailVerified: boolean | undefined;
    isAnonymous: boolean | undefined;
    tenantId: string | null | undefined;
    providerInfo: {
      providerId: string;
      displayName: string | null;
      email: string | null;
      photoUrl: string | null;
    }[];
  }
}

export function handleFirestoreError(error: unknown, operationType: OperationType, path: string | null) {
  const errInfo: FirestoreErrorInfo = {
    error: error instanceof Error ? error.message : String(error),
    authInfo: {
      userId: auth.currentUser?.uid,
      email: auth.currentUser?.email,
      emailVerified: auth.currentUser?.emailVerified,
      isAnonymous: auth.currentUser?.isAnonymous,
      tenantId: auth.currentUser?.tenantId,
      providerInfo: auth.currentUser?.providerData.map(provider => ({
        providerId: provider.providerId,
        displayName: provider.displayName,
        email: provider.email,
        photoUrl: provider.photoURL
      })) || []
    },
    operationType,
    path
  };
  console.error('Firestore Error: ', JSON.stringify(errInfo));
  throw new Error(JSON.stringify(errInfo));
}

// Connection test
async function testConnection() {
  try {
    await getDocFromServer(doc(db, 'test', 'connection'));
  } catch (error) {
    if (error instanceof Error && error.message.includes('the client is offline')) {
      console.error("Please check your Firebase configuration.");
    }
  }
}
testConnection();
