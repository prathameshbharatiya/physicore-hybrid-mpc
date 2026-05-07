import { initializeApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider } from 'firebase/auth';
import { getFirestore, doc, setDoc, getDocFromServer } from 'firebase/firestore';

const configFiles = import.meta.glob('../firebase-applet-config.json', { eager: true });
const firebaseConfig = (configFiles['../firebase-applet-config.json'] as any)?.default || {
  projectId: "gen-lang-client-0400823843",
  appId: "1:1026415005415:web:cf6d1b2f38a0c7ac1f9c42",
  apiKey: "AIzaSyBjvw8ewBhuVhjqy86LqeNNTw-egmdAXW8",
  authDomain: "gen-lang-client-0400823843.firebaseapp.com",
  storageBucket: "gen-lang-client-0400823843.firebasestorage.app",
  messagingSenderId: "1026415005415",
};

const config = {
  apiKey:            import.meta.env.VITE_FIREBASE_API_KEY            || firebaseConfig.apiKey,
  authDomain:        import.meta.env.VITE_FIREBASE_AUTH_DOMAIN        || firebaseConfig.authDomain,
  projectId:         import.meta.env.VITE_FIREBASE_PROJECT_ID         || firebaseConfig.projectId,
  storageBucket:     import.meta.env.VITE_FIREBASE_STORAGE_BUCKET     || firebaseConfig.storageBucket,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID|| firebaseConfig.messagingSenderId,
  appId:             import.meta.env.VITE_FIREBASE_APP_ID             || firebaseConfig.appId,
};

const app  = initializeApp(config);
export const auth          = getAuth(app);
export const db            = getFirestore(app);
export const googleProvider = new GoogleAuthProvider();

// Export the project ID so App.tsx can show it in error messages
export const FIREBASE_PROJECT_ID = config.projectId;

// Runs once on startup — logs whether Firestore is reachable
// Check browser console to see if this says OK or FAILED
getDocFromServer(doc(db, 'test', 'connection'))
  .then(() => console.log(`[FIRESTORE] Connected OK — project: ${config.projectId}`))
  .catch(err => console.error(`[FIRESTORE] Connection FAILED — project: ${config.projectId} — error: ${err.code} ${err.message}. Go to Firebase Console → Firestore → Rules and publish the security rules.`));

export enum OperationType {
  CREATE = 'create', UPDATE = 'update', DELETE = 'delete',
  LIST = 'list', GET = 'get', WRITE = 'write',
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
  };
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
        photoUrl: provider.photoURL,
      })) || [],
    },
    operationType,
    path,
  };
  console.error('Firestore Error:', JSON.stringify(errInfo));
  throw new Error(JSON.stringify(errInfo));
}
