/*
 * chat.m — Interactive TUI chat client for Flash-MoE inference server
 *
 * Thin HTTP/SSE client with session persistence.
 * Conversations saved to ~/.flash-moe/sessions/<session_id>.jsonl
 * Resume with: ./chat --resume <session_id>
 *
 * Build:  make chat
 * Run:    ./chat [--port 8000] [--show-think] [--resume <id>]
 */

#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <getopt.h>
#include <dirent.h>

#define MAX_INPUT_LINE 4096
#define MAX_RESPONSE (1024 * 1024)
#define SESSIONS_DIR_BASE ".flash-moe/sessions"

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static int json_escape(const char *src, char *buf, int bufsize) {
    int j = 0;
    for (int i = 0; src[i] && j < bufsize - 6; i++) {
        switch (src[i]) {
            case '"':  buf[j++]='\\'; buf[j++]='"'; break;
            case '\\': buf[j++]='\\'; buf[j++]='\\'; break;
            case '\n': buf[j++]='\\'; buf[j++]='n'; break;
            case '\r': buf[j++]='\\'; buf[j++]='r'; break;
            case '\t': buf[j++]='\\'; buf[j++]='t'; break;
            default:   buf[j++]=src[i]; break;
        }
    }
    buf[j] = 0;
    return j;
}

// ============================================================================
// Session persistence
// ============================================================================

static char g_sessions_dir[1024];

static void init_sessions_dir(void) {
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(g_sessions_dir, sizeof(g_sessions_dir), "%s/%s", home, SESSIONS_DIR_BASE);
    mkdir(g_sessions_dir, 0755);
    // Also create parent
    char parent[1024];
    snprintf(parent, sizeof(parent), "%s/.flash-moe", home);
    mkdir(parent, 0755);
    mkdir(g_sessions_dir, 0755);
}

static void session_path(const char *session_id, char *path, size_t pathsize) {
    snprintf(path, pathsize, "%s/%s.jsonl", g_sessions_dir, session_id);
}

// Append a turn to the session JSONL file
static void session_save_turn(const char *session_id, const char *role, const char *content) {
    char path[1024];
    session_path(session_id, path, sizeof(path));
    FILE *f = fopen(path, "a");
    if (!f) return;
    char escaped[MAX_RESPONSE * 2];
    json_escape(content, escaped, sizeof(escaped));
    fprintf(f, "{\"role\":\"%s\",\"content\":\"%s\"}\n", role, escaped);
    fclose(f);
}

// Load session history and replay to screen
static int session_load(const char *session_id) {
    char path[1024];
    session_path(session_id, path, sizeof(path));
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    printf("[resuming session %s]\n\n", session_id);
    int turns = 0;
    char line[MAX_RESPONSE];
    while (fgets(line, sizeof(line), f)) {
        // Simple parsing: find role and content
        char *role_start = strstr(line, "\"role\":\"");
        char *content_start = strstr(line, "\"content\":\"");
        if (!role_start || !content_start) continue;

        role_start += 8;
        char role[32]; int ri = 0;
        while (*role_start && *role_start != '"' && ri < 31) role[ri++] = *role_start++;
        role[ri] = 0;

        content_start += 11;
        // Decode the content (unescape)
        char content[MAX_RESPONSE]; int ci = 0;
        for (int i = 0; content_start[i] && ci < MAX_RESPONSE - 1; i++) {
            // Stop at closing quote (not escaped)
            if (content_start[i] == '"' && (i == 0 || content_start[i-1] != '\\')) break;
            if (content_start[i] == '\\' && content_start[i+1]) {
                i++;
                switch (content_start[i]) {
                    case 'n': content[ci++] = '\n'; break;
                    case 't': content[ci++] = '\t'; break;
                    case '"': content[ci++] = '"'; break;
                    case '\\': content[ci++] = '\\'; break;
                    default: content[ci++] = content_start[i]; break;
                }
            } else {
                content[ci++] = content_start[i];
            }
        }
        content[ci] = 0;

        if (strcmp(role, "user") == 0) {
            printf("\033[1m> %s\033[0m\n\n", content);
        } else if (strcmp(role, "assistant") == 0) {
            printf("%s\n\n", content);
        }
        turns++;
    }
    fclose(f);
    if (turns > 0) printf("[%d turns loaded]\n\n", turns);
    return turns;
}

// List recent sessions
static void session_list(void) {
    DIR *dir = opendir(g_sessions_dir);
    if (!dir) { printf("No sessions found.\n\n"); return; }

    printf("Recent sessions:\n");
    struct dirent *entry;
    int count = 0;
    while ((entry = readdir(dir))) {
        if (entry->d_name[0] == '.') continue;
        char *dot = strrchr(entry->d_name, '.');
        if (!dot || strcmp(dot, ".jsonl") != 0) continue;
        *dot = 0; // strip .jsonl

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s.jsonl", g_sessions_dir, entry->d_name);
        struct stat st;
        stat(path, &st);

        // Count lines (turns)
        FILE *f = fopen(path, "r");
        int lines = 0;
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) lines++;
            fclose(f);
        }

        printf("  %s  (%d turns)\n", entry->d_name, lines);
        count++;
    }
    closedir(dir);
    if (count == 0) printf("  (none)\n");
    printf("\n");
}

// ============================================================================
// HTTP / SSE
// ============================================================================

static void generate_session_id(char *buf, size_t bufsize) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    snprintf(buf, bufsize, "chat-%d-%ld%06d",
             (int)getpid(), (long)tv.tv_sec, (int)tv.tv_usec);
}

static int send_chat_request(int port, const char *user_message, int max_tokens, const char *session_id) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return -1; }

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "\n[error] Cannot connect to server on port %d.\n", port);
        close(sock);
        return -1;
    }

    char escaped[MAX_INPUT_LINE * 2];
    json_escape(user_message, escaped, sizeof(escaped));

    char body[MAX_INPUT_LINE * 3];
    int body_len = snprintf(body, sizeof(body),
        "{\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],"
        "\"max_tokens\":%d,\"stream\":true,\"session_id\":\"%s\"}",
        escaped, max_tokens, session_id);

    char request[MAX_INPUT_LINE * 4];
    int req_len = snprintf(request, sizeof(request),
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost:%d\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s",
        port, body_len, body);

    write(sock, request, req_len);
    return sock;
}

// Stream SSE response, accumulate text, return malloc'd response string
static char *stream_response(int sock, int show_thinking) {
    FILE *stream = fdopen(sock, "r");
    if (!stream) { close(sock); return NULL; }

    int header_done = 0, in_think = 0, tokens = 0;
    double t_start = now_ms(), t_first = 0;

    char *response = calloc(1, MAX_RESPONSE);
    int resp_len = 0;

    char line[65536];
    while (fgets(line, sizeof(line), stream)) {
        if (!header_done) {
            if (strcmp(line, "\r\n") == 0 || strcmp(line, "\n") == 0) header_done = 1;
            continue;
        }
        if (strncmp(line, "data: ", 6) != 0) continue;
        if (strncmp(line + 6, "[DONE]", 6) == 0) break;

        char *ck = strstr(line + 6, "\"content\":\"");
        if (!ck) continue;
        ck += 11;

        char decoded[4096]; int di = 0;
        for (int i = 0; ck[i] && ck[i] != '"' && di < 4095; i++) {
            if (ck[i] == '\\' && ck[i+1]) {
                i++;
                switch (ck[i]) {
                    case 'n': decoded[di++]='\n'; break;
                    case 't': decoded[di++]='\t'; break;
                    case '"': decoded[di++]='"'; break;
                    case '\\': decoded[di++]='\\'; break;
                    default: decoded[di++]=ck[i]; break;
                }
            } else decoded[di++] = ck[i];
        }
        decoded[di] = 0;
        if (!di) continue;

        if (strstr(decoded, "<think>")) in_think = 1;
        if (strstr(decoded, "</think>")) { in_think = 0; tokens++; continue; }
        tokens++;
        if (!t_first) t_first = now_ms();

        // Accumulate non-thinking response
        if (!in_think && resp_len + di < MAX_RESPONSE - 1) {
            memcpy(response + resp_len, decoded, di);
            resp_len += di;
            response[resp_len] = 0;
        }

        if (in_think && !show_thinking) continue;
        if (in_think) printf("\033[2m%s\033[0m", decoded);
        else printf("%s", decoded);
        fflush(stdout);
    }
    fclose(stream);

    double gen_time = t_first > 0 ? now_ms() - t_first : 0;
    int gen_tokens = tokens > 1 ? tokens - 1 : 0;
    printf("\n\n");
    if (gen_tokens > 0 && gen_time > 0)
        printf("[%d tokens, %.1f tok/s, TTFT %.1fs]\n\n",
               tokens, gen_tokens * 1000.0 / gen_time,
               t_first > 0 ? (t_first - now_ms() + gen_time + (t_first - (now_ms() - gen_time))) / 1000.0 : 0);

    return response;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    int port = 8000;
    int max_tokens = 8192;
    int show_thinking = 0;
    const char *resume_id = NULL;

    static struct option long_options[] = {
        {"port",        required_argument, 0, 'p'},
        {"max-tokens",  required_argument, 0, 't'},
        {"show-think",  no_argument,       0, 's'},
        {"resume",      required_argument, 0, 'r'},
        {"sessions",    no_argument,       0, 'l'},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    init_sessions_dir();

    int c;
    while ((c = getopt_long(argc, argv, "p:t:sr:lh", long_options, NULL)) != -1) {
        switch (c) {
            case 'p': port = atoi(optarg); break;
            case 't': max_tokens = atoi(optarg); break;
            case 's': show_thinking = 1; break;
            case 'r': resume_id = optarg; break;
            case 'l': session_list(); return 0;
            case 'h':
                printf("Usage: %s [options]\n", argv[0]);
                printf("  --port N         Server port (default: 8000)\n");
                printf("  --max-tokens N   Max response tokens (default: 8192)\n");
                printf("  --show-think     Show <think> blocks (dimmed)\n");
                printf("  --resume ID      Resume a previous session\n");
                printf("  --sessions       List saved sessions\n");
                printf("  --help           This message\n");
                return 0;
            default: return 1;
        }
    }

    char session_id[64];
    if (resume_id) {
        strncpy(session_id, resume_id, sizeof(session_id) - 1);
        session_id[sizeof(session_id) - 1] = 0;
    } else {
        generate_session_id(session_id, sizeof(session_id));
    }

    printf("==================================================\n");
    printf("  Qwen3.5-397B-A17B Chat (Flash-MoE)\n");
    printf("==================================================\n");
    printf("  Server:  http://localhost:%d\n", port);
    printf("  Session: %s%s\n", session_id, resume_id ? " (resumed)" : "");
    printf("\n  Commands: /quit /exit /clear /sessions\n");
    printf("==================================================\n\n");

    // Health check
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "Server not running on port %d.\n", port);
        fprintf(stderr, "Start it: ./infer --serve %d\n\n", port);
        close(sock);
        return 1;
    }
    close(sock);

    // Resume: load and display previous conversation
    if (resume_id) {
        int turns = session_load(session_id);
        if (turns == 0) {
            printf("No session found with ID: %s\n\n", session_id);
        }
        // Note: server-side KV cache may not match if server restarted.
        // The conversation will continue but model won't "remember" old context
        // unless we re-prefill (TODO: detect server restart and replay).
    }

    printf("Ready to chat.\n\n");

    char input_line[MAX_INPUT_LINE];

    for (;;) {
        printf("> ");
        fflush(stdout);

        if (!fgets(input_line, sizeof(input_line), stdin)) {
            printf("\n");
            break;
        }

        size_t len = strlen(input_line);
        while (len > 0 && (input_line[len-1] == '\n' || input_line[len-1] == '\r'))
            input_line[--len] = 0;

        if (len == 0) continue;
        if (strcmp(input_line, "/quit") == 0 || strcmp(input_line, "/exit") == 0) {
            printf("Goodbye.\n");
            break;
        }
        if (strcmp(input_line, "/clear") == 0) {
            generate_session_id(session_id, sizeof(session_id));
            printf("[new session: %s]\n\n", session_id);
            continue;
        }
        if (strcmp(input_line, "/sessions") == 0) {
            session_list();
            continue;
        }

        // Save user turn
        session_save_turn(session_id, "user", input_line);

        sock = send_chat_request(port, input_line, max_tokens, session_id);
        if (sock < 0) continue;

        printf("\n");
        char *response = stream_response(sock, show_thinking);

        // Save assistant turn
        if (response && strlen(response) > 0) {
            session_save_turn(session_id, "assistant", response);
        }
        free(response);
    }

    return 0;
}
