
/**
 * Simple hash function for checksum validation.
 * Returns a 4-character hex string.
 */
export const simpleHash = (str: string): string => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash).toString(16).substring(0, 4).toUpperCase().padStart(4, '0');
};

/**
 * Generates a unique project ID.
 * Format: [EMAIL_HASH_4]-[RANDOM_HEX_4]
 */
export const generateId = (email?: string): string => {
  const emailHash = email ? simpleHash(email) : 'ANON';
  const randomHex = Math.random().toString(16).substring(2, 6).toUpperCase();
  return `${emailHash}-${randomHex}`;
};

/**
 * Encodes a project payload into a Project Code string.
 * Format: PC-[BASE64_DATA]
 */
export const encodeProjectCode = (payload: any): string => {
  try {
    const jsonStr = JSON.stringify(payload);
    const base64 = btoa(encodeURIComponent(jsonStr));
    return `PC-${base64}`;
  } catch (error) {
    console.error("Encoding error:", error);
    return "";
  }
};

/**
 * Decodes a Project Code string into a project payload.
 */
export const decodeProjectCode = (code: string): any => {
  try {
    if (!code.startsWith('PC-') && !code.startsWith('SN-')) {
      throw new Error("Invalid Project Code prefix");
    }
    const base64 = code.substring(3);
    const jsonStr = decodeURIComponent(atob(base64));
    const payload = JSON.parse(jsonStr);

    // Validate checksum
    const { checksum, ...rest } = payload;
    const calculatedChecksum = simpleHash(JSON.stringify(rest));
    
    if (checksum !== calculatedChecksum) {
      throw new Error("Checksum mismatch: Data may be corrupted");
    }

    return payload;
  } catch (error) {
    console.error("Decoding error:", error);
    return null;
  }
};
