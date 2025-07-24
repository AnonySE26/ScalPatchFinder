from gritlm import GritLM

tokenizer = GritLM("GritLM/GritLM-7B", torch_dtype="auto").tokenizer

diff = """diff --git a/java/org/apache/tomcat/util/net/openssl/LocalStrings.properties b/java/org/apache/tomcat/util/net/openssl/LocalStrings.properties
index 84990f3d8eee..34ec880c412e 100644
--- a/java/org/apache/tomcat/util/net/openssl/LocalStrings.properties
+++ b/java/org/apache/tomcat/util/net/openssl/LocalStrings.properties
@@ -17,6 +17,7 @@ engine.ciphersFailure=Failed getting cipher list
 engine.emptyCipherSuite=Empty cipher suite
 engine.engineClosed=Engine is closed
 engine.failedCipherSuite=Failed to enable cipher suite [{0}]
+engine.failedToReadAvailableBytes=There are plain text bytes available to read but no bytes were read
 engine.inboundClose=Inbound closed before receiving peer's close_notify
 engine.invalidBufferArray=offset: [{0}], length: [{1}] (expected: offset <= offset + length <= srcs.length [{2}])
 engine.invalidDestinationBuffersState=The state of the destination buffers changed concurrently while unwrapping bytes
diff --git a/java/org/apache/tomcat/util/net/openssl/OpenSSLEngine.java b/java/org/apache/tomcat/util/net/openssl/OpenSSLEngine.java
index 59c1d5f009fd..4700c2a4b254 100644
--- a/java/org/apache/tomcat/util/net/openssl/OpenSSLEngine.java
+++ b/java/org/apache/tomcat/util/net/openssl/OpenSSLEngine.java
@@ -591,8 +591,10 @@ public synchronized SSLEngineResult unwrap(final ByteBuffer src, final ByteBuffe
                     throw new SSLException(e);
                 }
 
-                if (bytesRead == 0) {
-                    break;
+                if (bytesRead <= 0) {
+                    // This should not be possible. pendingApp is positive
+                    // therefore the read should have read at least one byte.
+                    throw new IllegalStateException(sm.getString("engine.failedToReadAvailableBytes"));
                 }
 
                 bytesProduced += bytesRead;
diff --git a/webapps/docs/changelog.xml b/webapps/docs/changelog.xml
index 64eefbbc9ed3..370572ec6258 100644
--- a/webapps/docs/changelog.xml
+++ b/webapps/docs/changelog.xml
@@ -185,6 +185,10 @@
         fully cleared, as there could be more than one error present after
         an operation (confirmed in the OpenSSL API documentation). (remm)
       </fix>
+      <fix>
+        Make handling of OpenSSL read errors more robust when plain text data is
+        reported to be available to read. (markt)
+      </fix>
     </changelog>
   </subsection>
   <subsection name="Web applications">"""

print(len(tokenizer.tokenize(diff)), flush=True)
